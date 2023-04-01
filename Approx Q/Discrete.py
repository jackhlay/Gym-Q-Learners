import gym
import numpy as np

games= ['CartPole-v1','MountainCar-v0', 'Acrobot-v1']
env = gym.make('CartPole-v1')

# define the function approximator
def linear_approx(state, weights):
    return np.dot(state, weights)

# define hyperparameters
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

# initialize weights
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
weights = np.random.rand(num_states, num_actions)

# define training loop
episodes = 1024
scores = []
for ep in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        # choose action
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = linear_approx(state, weights)
            action = np.argmax(q_values)

        # take action and observe next state and reward
        next_state, reward, done, _ = env.step(action)

        # update weights
        if done:
            target = reward
        else:
            next_q_values = linear_approx(next_state, weights)
            target = reward + gamma * np.max(next_q_values)
        q_values = linear_approx(state, weights)
        q_values[action] = (1 - alpha) * q_values[action] + alpha * target
        weights += np.outer(state, q_values - linear_approx(state, weights))

        # update state and score
        state = next_state
        score += reward

    # update epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # print progress
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print(f"Episode {ep}: Score {score}, Avg Score {avg_score:.2f}")

# test final policy
state = env.reset()
done = False
score = 0
while not done:
    env.render()
    q_values = linear_approx(state, weights)
    action = np.argmax(q_values)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    score += reward
print(f"Final Score: {score}")
env.close()
