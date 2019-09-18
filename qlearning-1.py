import gym
import numpy as np

env = gym.make('MountainCar-v0')

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25_000

SHOW_EVERY = 2000

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
discrete_obs_win_size = (env.observation_space.high-env.observation_space.low) / DISCRETE_OBS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low)/ discrete_obs_win_size
	return tuple(discrete_state.astype(np.int))

hit_hist = []
reward_hist = []
for episode in range(EPISODES):
	discrete_state = get_discrete_state(env.reset())
	if episode % 500 == 0 and episode>0:
			proportion_won = sum(hit_hist[-21:-1])/len(hit_hist[-21:-1])
			avg_reward = np.average(reward_hist[-21:-1])
			print('Success {}% of the time with average reward {} in episode {}.'.format(round(proportion_won*100,1), avg_reward, episode))
	done = False
	cummulative_reward = 0
	while not done:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		new_state, reward, done, _ = env.step(action)
		cummulative_reward += reward
		new_discrete_state = get_discrete_state(new_state)
		if episode % SHOW_EVERY == 0:
			env.render()
		
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]

			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action,)] = new_q
		elif new_state[0] >= env.goal_position:
			q_table[discrete_state + (action,)] = 0
			hit_hist.append(1)
		else:
			hit_hist.append(0)

		discrete_state = new_discrete_state
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value
	reward_hist.append(cummulative_reward)
env.env.close()