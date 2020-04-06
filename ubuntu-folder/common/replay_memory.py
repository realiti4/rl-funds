import numpy as np




class Replay_Memory():
    def __init__(self):
        self.memory_size = 100000
        self.memory = np.zeros(self.memory_size)

        self.np_state = np.zeros([self.memory_size, 4], dtype=np.float32)
        self.np_next_state = np.zeros([self.memory_size, 4], dtype=np.float32)
        self.np_action = np.zeros(self.memory_size, dtype=np.int8)
        self.np_reward = np.zeros(self.memory_size, dtype=np.float32)
        self.np_done = np.full((self.memory_size), True)

    def push(self, state, action, reward, next_state, done):
        index = steps_done % self.memory_size

        self.np_state[index] = state
        self.np_action[index] = action
        self.np_reward[index] = reward
        self.np_next_state[index] = next_state
        self.np_done[index] = done


    def sample(self, batch_size):
        mask = np.random.randint(self.memory_size, size=batch_size)

        sample = [self.np_state[mask], self.np_action[mask], self.np_reward[mask], self.np_next_state[mask], self.np_done[mask]]


        return sample[0], sample[1], sample[2], sample[3], sample[4]

    def __len__(self):
        return steps_done