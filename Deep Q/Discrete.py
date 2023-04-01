import gym
import random
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten

games= ['CartPole-v1','MountainCar-v0', 'Acrobot-v1']
env = gym.make('CartPole-v1')

states = env.observation_space.shape
actions = env.action_space.n

episodes = 512
scores=[]
for ep in range(1,episodes+1):
    state= env.reset()
    done= False
    score = 0

    while not done:
        env.render()
        action = random.choice([0,1])
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(ep,score))
    scores.append(score)
avg_score = np.mean(scores)
print(f"Average score per episode: {avg_score}, Max score: {max(scores)}")

def build_model(states, actions):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(1 ,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

from keras.optimizers import Adam

model = build_model(states, actions)

model.compile(loss='mse', optimizer=Adam(lr=1e-3), metrics=['mse'])

# define hyperparameters
gamma = 1  # discount factor
epsilon = .99  # exploration rate
epsilon_min = 0.01  # minimum exploration rate
epsilon_decay = 0.995
batch_size = 256  # minibatch size
replay_memory = []  # replay memory

# train model
state = env.reset()
done = False
steps = 0
while not done:
    # choose action
    if np.random.rand() <= epsilon:
        action = env.action_space.sample()
    else:
        q_values = model.predict(np.array([state]))
        action = np.argmax(q_values[0])

    # take action and observe next state and reward
    next_state, reward, done, _ = env.step(action)

    # store experience in replay memory
    # (s, a, r, s', done)
    experience = (state, action, reward, next_state, done)
    replay_memory.append(experience)

    # update state
    state = next_state
    steps += 1

    # update epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # sample random minibatch from replay memory
    if len(replay_memory) >= batch_size:
        minibatch = random.sample(replay_memory, batch_size)

        # compute target Q-values
        X = []
        y = []
        for state, action, reward, next_state, done in minibatch:
            q_values = model.predict(np.array([state]))
            if done:
                q_values[0][action] = reward
            else:
                next_q_values = model.predict(np.array([next_state]))
                q_values[0][action] = reward + gamma * np.max(next_q_values)

            X.append(state)
            y.append(q_values)

        # fit model on minibatch
        model.fit(np.array(X), np.array(y), batch_size=len(X), verbose=0)
env.close()


