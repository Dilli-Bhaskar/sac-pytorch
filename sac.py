import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from .memory import ReplayMemory, Transition

# Get device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Agent:

    def __init__(
            self,
            state_dim,
            action_dim,
            alpha,
            lr=3e-4,
            discount=0.99,
            tau=0.005,
            log_std_min=-20,
            log_std_max=2,
            memory_size=1000000,
            batch_size=256):
        self.alpha = alpha
        self.discount = discount
        self.tau = tau
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.normal = Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
        # SAC has five networks
        # Two Q Networks, One V Network, One Target-V Network and One Policy Network
        self.q_net_1 = QNetwork(state_dim, action_dim).to(device)
        self.q_net_1_optimizer = optim.Adam(self.q_net_1.parameters(), lr=lr)
        self.q_net_2 = QNetwork(state_dim, action_dim).to(device)
        self.q_net_2_optimizer = optim.Adam(self.q_net_2.parameters(), lr=lr)
        self.v_net = VNetwork(state_dim).to(device)
        self.v_net_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        self.target_v_net = VNetwork(state_dim).to(device)
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(param.data)
        self.policy_net = PolicyNetwork(state_dim, action_dim, log_std_min, log_std_max, self.normal).to(device)
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = self.policy_net.predict(state)
        return action.cpu()[0].detach().numpy()

    def get_rollout_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, _ = self.policy_net.forward(state)
        return mean.tanh().cpu()[0].detach().numpy()

    def store_transition(self, state, action, reward, next_state, end):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = torch.FloatTensor(action).unsqueeze(0).to(device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        end = torch.FloatTensor([end]).unsqueeze(0).to(device)
        self.memory.push(state, action, reward, next_state, end)

    def learn(self):
        # Sample experiences
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state)
        end = torch.cat(batch.end)

        # Training Q Networks
        predicted_q_value_1 = self.q_net_1.forward(state, action)
        predicted_q_value_2 = self.q_net_2.forward(state, action)
        predicted_v_target = self.target_v_net.forward(next_state)
        
        
        target_q_value = reward + (1 - end) * self.discount * predicted_v_target
        
        q_loss_1 = nn.MSELoss()(predicted_q_value_1, target_q_value.detach())
        q_loss_2 = nn.MSELoss()(predicted_q_value_2, target_q_value.detach())
        self.q_net_1_optimizer.zero_grad()
        q_loss_1.backward()
        self.q_net_1_optimizer.step()
        self.q_net_2_optimizer.zero_grad()
        q_loss_2.backward()
        self.q_net_2_optimizer.step()

        # Training V Network
        predicted_v_value = self.v_net.forward(state)
        new_action, log_prob = self.policy_net.predict(state)
        predicted_new_q_value_1 = self.q_net_1.forward(state, new_action)
        predicted_new_q_value_2 = self.q_net_2.forward(state, new_action)
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
        target_v_value = predicted_new_q_value - self.alpha * log_prob
        v_loss = nn.MSELoss()(predicted_v_value, target_v_value.detach())
        self.v_net_optimizer.zero_grad()
        v_loss.backward()
        self.v_net_optimizer.step()

        # Training Policy Network
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()

        # Updating Target-V Network
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class VNetwork(nn.Module):

    def __init__(self, state_dim):
        super(VNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, log_std_min, log_std_max, normal):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.normal = normal

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)

        self.mean_layer3 = nn.Linear(256, action_dim)
        self.log_std_layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        mean = self.mean_layer3(x)
        log_std = self.log_std_layer3(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        z = self.normal.sample(sample_shape=std.shape)
        action_raw = mean + std * z
        action = torch.tanh(action_raw)
        log_prob = Normal(mean, std).log_prob(action_raw) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob



