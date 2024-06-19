import sys
import sac
import sac_cepo
from utils import get_normalized_env
import gym


# py -3 train.py sac Pendulum-v0 5 10000 test.csv

def train():
    agent_name = sac

    env = 'HalfCheetah-v4'
    reward_scale = 5
    max_step = 1e6
    evaluate_step = 1000
    output_file = 'test.csv'
    print("Agent: {}".format(agent_name))
    # Load environment and agent
    env = gym.make('HalfCheetah-v4')
    eval_env =  gym.make('HalfCheetah-v4')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = sac.Agent(env=env,state_dim=state_dim, action_dim=action_dim, alpha=1 / reward_scale)


    # Initial exploration for 1000 steps
    step = 0
    while True:
        state = env.reset()[0]
        while True:
            action = env.action_space.sample()
            next_state, reward, end,done ,_ = env.step(action)
            step += 1
            agent.store_transition(state, action, reward, next_state, end * 1)
            state = next_state
            if end or done:
                break
            if step == 1000:
                break
        if step == 1000:
            break

    # Formal training
    step = 0
    while True:
        state = env.reset()[0]
        while True:
            action = agent.choose_action(state)
            next_state, reward, end,done, _ = env.step(action)
            step += 1
            agent.store_transition(state, action, reward, next_state, end * 1)
            agent.learn()
            state = next_state
            if step % evaluate_step == 0:
                evaluate_reward = rollout(agent, eval_env)
                print(step, evaluate_reward)
                with open(output_file, "a") as file:
                    file.write("{},{}\n".format(step, evaluate_reward))
            if end or done:
                break
            if step == max_step:
                break
        if step == max_step:
            break

    env.close()
    eval_env.close()
    print("Training finished.")


def rollout(agent, env):
    average_reward = 0
    for i in range(10):
        state = env.reset()[0]
        total_reward = 0
        while True:
            action = agent.get_rollout_action(state)
            next_state, reward, end,done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if end or done:
                break
        average_reward += total_reward
    return average_reward / 10


if __name__ == "__main__":
    train()
