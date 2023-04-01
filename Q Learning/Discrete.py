import gym
import numpy as np

# Can only run cartpole right now :(
env = gym.make('CartPole-v0')

# Initialize Q-table
states = np.linspace(-1, 1, num=20) # discretize the state space
actions = [0, 1]
q_table = np.zeros((len(states), len(states), len(actions)))

# Set hyperparameters
learning_rate = 0.001 # alpha
discount_factor = 0.9 # gamma
epsilon = 1
epsilon_decay = 0.95
min_epsilon = 0.001
num_episodes = 2000

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
    avg_score = np.mean(scores[-100:])
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
