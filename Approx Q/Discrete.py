import gym
import numpy as np
import matplotlib.pyplot as plt

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
episodes = 10000
scores = []
avgs= []
maxx=0
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
        if maxx<score:
            maxx=score
        else:
            maxx=maxx

    # update epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # print progress
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    avgs.append(avg_score)
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
print(f"Final Avg: {avg_score:.2f}, max score: {maxx}")
env.close()

#normalize avg scores
avgs= np.array(avgs[::100])
xax = np.array(range(len(avgs)))
slope, intercept = np.polyfit(xax, avgs, 1)
trendline_x = np.array([np.min(xax), np.max(xax)])
trendline_y = slope * trendline_x + intercept

fig, ax = plt.subplots()
ax.plot(trendline_x, trendline_y, color='red', linestyle=':')
ax.plot(xax, avgs, color='steelblue', linestyle='-.', marker='o')
ax.set_xlabel('Attempts (x100)')
ax.set_ylabel('Avg Score')
ax.set_title('Average Score/Time')
plt.show()
