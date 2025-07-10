import numpy as np
from collections import defaultdict

class GridWorld:
    def __init__(self, size = 4, goal = (3, 3)):
        self.size = size
        self.goal = goal
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos
    
    def step(self, action):
        row, col = self.agent_pos

        if action == 'up':
            intended_pos = (row - 1, col)
        elif action == 'down':
            intended_pos = (row + 1, col)
        elif action == 'left':
            intended_pos = (row, col - 1)
        elif action == 'right':
            intended_pos = (row, col + 1)
        else:
            raise ValueError("Invalid action")

        if 0 <= intended_pos[0] < self.size and 0 <= intended_pos[1] < self.size:
            self.agent_pos = intended_pos
            reward = 1 if self.agent_pos == self.goal else 0
            done = self.agent_pos == self.goal
        else:
            reward = -1
            done = False

        return self.agent_pos, reward, done
    
    def q_learning(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon = 1):

        q_table = defaultdict(lambda: np.zeros(4))

        actions = ['up', 'down', 'left', 'right']
        action_to_index = {a: i for i, a in enumerate(actions)}
        index_to_action = {i: a for i, a in enumerate(actions)}

        for episode in range(episodes):
            state = self.reset()
            done = False
            while not done:
                if np.random.rand() < epsilon:
                    action_index = np.random.choice(4)
                else:
                    action_index = np.argmax(q_table[state])

                action = index_to_action[action_index]

                next_state, reward, done = self.step(action)

                best_next_action = np.max(q_table[next_state])
                td_target = reward + gamma * best_next_action
                td_error = td_target - q_table[state][action_index]
                q_table[state][action_index] += alpha * td_error

                state = next_state

            epsilon = max(0.01, epsilon * 0.995)

        return q_table
    
def get_optimal_path(env, q_table):
    state = env.reset()
    path = [state]
    visited = set()
    
    for _ in range(100):
        action_index = np.argmax(q_table[state])
        action = ['up', 'down', 'left', 'right'][action_index]
        next_state, reward, done = env.step(action)

        if next_state in visited:
            break

        path.append(next_state)
        visited.add(next_state)
        state = next_state

        if done:
            break

    return path



env = GridWorld()
q_table = env.q_learning()


optimal_path = get_optimal_path(env, q_table)
print("Best path from start to goal:")
for step in optimal_path:
    print(step)
