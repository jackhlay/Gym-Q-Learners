import gym
import numpy as np

# Initialize the environment
env = gym.make("CliffWalking-v0")
# Set the number of episodes to train for
num_episodes = 1000
# Set the learning rate
learning_rate = 0.8
# Set the discount factor
discount_factor = 0.95

# Initialize the Q table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Define a function to choose an action based on the Q table and exploration rate
def choose_action(state, exploration_rate):
    if np.random.uniform(0, 1) < exploration_rate:
        # Choose a random action
        action = env.action_space.sample()
    else:
        # Choose the action with the highest Q value
        action = np.argmax(Q[state, :])
    return action

# Loop over each episode
for episode in range(num_episodes):
    # Reset the environment and get the initial state
    state = env.reset()
    # Initialize the episode score
    score = 0
    # Initialize the exploration rate
    exploration_rate = 1.0 / (episode + 1)
    # Loop over each step in the episode
    while True:
        # Choose an action
        action = choose_action(state, exploration_rate)
        # Take the chosen action and observe the new state and reward
        new_state, reward, done, info = env.step(action)
        # Update the Q table
        Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])
        # Update the score
        score += reward
        # Update the state
        state = new_state
        # If the episode is over, break out of the loop
        if done:
            break
    # Print the score for the episode
    print("Episode {}: Score = {}".format(episode, score))

# Play the game using the learned Q table
state = env.reset()
while True:
    # Choose the action with the highest Q value
    action = np.argmax(Q[state, :])
    # Take the chosen action and observe the new state and reward
    new_state, reward, done, info = env.step(action)
    # Update the state
    state = new_state
    # Render the environment
    env.render()
    # If the game is over, break out of the loop
    if done:
        break

# Close the environment
env.close()
