import gym
import gym_maze
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.patches import Polygon
import matplotlib.image as mpimg

# Create an environment
env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()


def exploration_exploitation(state, epsilon, q_table):
    random = np.random.rand()
    if random < epsilon:  # exploration
        action = env.action_space.sample()
    else:  # exploitation
        x, y = state
        action = np.argmax(q_table[int(x), int(y), :])
    return action


def update_q_table(q_table, cur_state, action, next_state, reward, alpha, gamma):
    cur_x, cur_y = cur_state
    next_x, next_y = next_state
    # Q(S, A) <- (1 - α) Q(S, A) + [α * (r + (γ * max(Q(S', A*))))]
    sample = reward + (gamma * np.max(q_table[next_x, next_y, :]))
    q_table[int(cur_x), int(cur_y), action] += alpha * (sample - q_table[int(cur_x), int(cur_y), action])


def q_learning(num_episodes, q_table, epsilon, alpha, gamma, render_mode=False):
    actions = np.zeros((env.maze_size[0], env.maze_size[1]))

    episode_steps = []
    episode_rewards = []

    limit_steps = 500

    for i in range(num_episodes):

        # start from the initial state
        cur_state = env.reset()
        game_over = False

        num_actions = 0
        total_reward = 0

        while not game_over:

            env.render()
            # time.sleep(0.05)

            if num_actions > limit_steps:
                break

            # choose an action
            action = exploration_exploitation(cur_state, epsilon, q_table)
            if epsilon > 0 and num_actions != 0:
                epsilon -= ((0.00001 * i) / (num_actions))
            elif num_actions != 0:
                epsilon = 0

            num_actions += 1

            # Perform the action and receive feedback from the environment
            next_state, reward, done, truncated = env.step(action)

            if truncated:
                break

            if done:
                game_over = True

            else:
                total_reward += reward

            # update Q_table
            update_q_table(q_table, cur_state, action, next_state, reward, alpha, gamma)

            episode_steps.append(num_actions)
            episode_rewards.append(total_reward)

            # update state
            cur_state = next_state

        print("episode:", i, "  steps:", num_actions, "  reward:", total_reward, "  epsilon:", epsilon)

    return q_table, actions, episode_steps, episode_rewards


num_actions = env.action_space.n
num_row = env.maze_size[0]
num_column = env.maze_size[1]

# initialize Q-Table
Q_table = np.zeros((num_row, num_column, num_actions))

# Define the maximum number of iterations
NUM_EPISODES = 1000
# Define explroration explotation trade-off
epsilon = 1
# Define learning rate
alpha = 0.1
# Define discount factor
gamma = 0.9

# Q-Learning Algorithm
q_table, actions, episode_steps, episode_rewards = q_learning(NUM_EPISODES, Q_table, epsilon, alpha, gamma)


#-----------------------------------------------------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(episode_steps, label='Steps', color='blue')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Number of Steps', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2.plot(episode_rewards, label='Total Reward', color='green')
ax2.set_ylabel('Total Reward', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.show()


def show_q_table(q_table):
    rows, cols = q_table.shape[:2]

    fig, ax_array = plt.subplots(rows, cols, figsize=(cols, rows))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(rows):
        for j in range(cols):
            values = q_table[i, j, :]
            ax = ax_array[i, j]

            center = (0.5, 0.5)
            triangle1 = Polygon([
                [center[0] - 0.5, center[1] + 0.5],
                [center[0] + 0.5, center[1] + 0.5],
                [center[0], center[1]],
            ], closed=True, edgecolor='black', facecolor='none')

            triangle2 = Polygon([
                [center[0] - 0.5, center[1] + 0.5],
                [center[0] - 0.5, center[1] - 0.5],
                [center[0], center[1]],
            ], closed=True, edgecolor='black', facecolor='none')

            triangle3 = Polygon([
                [center[0] - 0.5, center[1] - 0.5],
                [center[0] + 0.5, center[1] - 0.5],
                [center[0], center[1]],
            ], closed=True, edgecolor='black', facecolor='none')

            triangle4 = Polygon([
                [center[0] + 0.5, center[1] - 0.5],
                [center[0] + 0.5, center[1] + 0.5],
                [center[0], center[1]],
            ], closed=True, edgecolor='black', facecolor='none')

            ax.text(center[0], center[1] + 0.25, f'{values[0]:.4f}', ha='center', va='center', fontsize=6)
            ax.text(center[0], center[1] - 0.25, f'{values[1]:.4f}', ha='center', va='center', fontsize=6)
            ax.text(center[0] + 0.25, center[1], f'{values[2]:.4f}', ha='center', va='center', fontsize=6)
            ax.text(center[0] - 0.25, center[1], f'{values[3]:.4f}', ha='center', va='center', fontsize=6)

            ax.add_patch(triangle1)
            ax.add_patch(triangle2)
            ax.add_patch(triangle3)
            ax.add_patch(triangle4)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_actions(actions, ax):
    actions = actions.astype(int)

    table = ax.table(cellText=actions, loc='center', cellLoc='center', edges='open')
    ax.axis('off')

    for i in range(actions.shape[0]):
        for j in range(actions.shape[1]):
            value = actions[i, j]
            if value == 0:
                arrow = u'$\u2191$'  # Up arrow
            elif value == 1:
                arrow = u'$\u2193$'  # Down arrow
            elif value == 2:
                arrow = u'$\u2192$'  # Right arrow
            elif value == 3:
                arrow = u'$\u2190$'  # Left arrow
            else:
                arrow = ''

            table[i, j].get_text().set_text(arrow)


# show results
show_q_table(q_table)

max_values = np.max(q_table, axis=2)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(max_values, cmap='viridis', interpolation='none')
axs[0].set_title('Q-table Visualization')
axs[0].axis('off')
for i in range(max_values.shape[0]):
    for j in range(max_values.shape[1]):
        axs[0].text(j, i, f'{max_values[i, j]:.2f}', ha='center', va='center', color='black')

show_actions(actions, axs[2])
axs[2].set_title('Actions')

image_path = 'env.PNG'
img = mpimg.imread(image_path)
axs[1].imshow(img)
axs[1].set_title('Image')
axs[1].axis('off')

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot
plt.show()


