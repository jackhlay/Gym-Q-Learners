import gym
import numpy as np
import time

start = time.time()
# Can only run cartpole right now :(
env = gym.make('CartPole-v0')

# Initialize Q-table
states = np.linspace(-1, 1, num=20) # discretize the state space
actions = [0, 1]
q_table = np.zeros((len(states), len(states), len(actions)))

# Set hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.01
num_episodes = 4200 #wanted to do 4196, but it plots data every 100 episodes

# Run Q-learning algorithm
scores = []
maxx = 0
for episode in range(num_episodes):
    state = env.reset()
    done = False
    score=0
    while not done:
        env.render()
        # Choose action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            state_idx = np.digitize(state, states) - 1
            q_values = q_table[state_idx[0], state_idx[1], :]
            action = np.argmax(q_values)

        # Take action and observe next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update Q-table
        next_state_idx = np.digitize(next_state, states) - 1
        q_values_next = q_table[next_state_idx[0], next_state_idx[1], :]
        td_target = reward + discount_factor * np.max(q_values_next)
        state_idx = np.digitize(state, states) - 1
        td_error = td_target - q_table[state_idx[0], state_idx[1], action]
        q_table[state_idx[0], state_idx[1], action] += learning_rate * td_error

        state = next_state
        score += reward
        if score > maxx:
            maxx = score
        else:
            maxx = maxx
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Save and print scores 
    scores.append(score)
    avg_score = np.mean(scores[-10:])
    print(f"Episode {episode+1}/{num_episodes} - Score: {score}, Avg. Score: {avg_score:.2f}, Max Score: {maxx}")

# Play the game using the learned Q-table
state = env.reset()
done = False
while not done:
    state_idx = np.digitize(state, states) - 1
    q_values = q_table[state_idx[0], state_idx[1], :]
    action = np.argmax(q_values)
    state, _, done, _ = env.step(action)
    env.render()
env.close()

import matplotlib.pyplot as plt

# Plot average score over time
avgs = []
for i in range(100, num_episodes+1, 100):
    avg = np.mean(scores[i-100:i])
    avgs.append(avg)

xax = np.arange(len(avgs))
slope, intercept = np.polyfit(xax, avgs, 1)
trendline_y = slope * xax + intercept

fig, ax = plt.subplots()
ax.plot(xax, trendline_y, linestyle='-.', color='goldenrod')
ax.plot(xax, avgs, linestyle='-', color='orangered', marker='o')
ax.set_xlabel('Episode (x100)')
ax.set_ylabel('Avg Score')
ax.set_title('Average Score/Time')
ax.set_facecolor("dimgray")
fig.set_facecolor("dimgray")
plt.show()

tot = time.time() - start
print(f"Time: {tot:.4f} seconds")