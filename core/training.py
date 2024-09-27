import numpy as np
import random
import matplotlib.pyplot as plt
from core.epsilon_decay import epsilon_decay, choose_action

def train_and_visualize(model, data, episodes, max_steps_per_episode, batch_size, gamma, epsilon_start, replay_memory_size):
    rewards_per_episode = []
    memory = []  # Replay memory to store experiences

    for episode in range(episodes):
        state, label = get_initial_state(data)
        total_reward = 0
        done = False
        step = 0
        epsilon = epsilon_decay(episode, initial_epsilon=epsilon_start)

        while not done and step < max_steps_per_episode:
            step += 1
            action = choose_action(state, epsilon, model, 3)
            next_state, reward, done = execute_action(action, state, step, max_steps_per_episode, data)
            total_reward += reward

            # Store transition in memory (state, action, reward, next_state, done)
            memory.append((state, action, reward, next_state, done))

            if len(memory) > replay_memory_size:
                memory.pop(0)

            # Train the model if we have enough experiences in memory
            if len(memory) >= batch_size:
                # Randomly sample a batch from memory
                batch = random.sample(memory, batch_size)
                states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

                states_b = np.array(states_b)
                actions_b = np.array(actions_b)
                rewards_b = np.array(rewards_b)
                next_states_b = np.array(next_states_b)
                dones_b = np.array(dones_b)

                for i in range(batch_size):
                    next_q_values = model.predict(np.expand_dims(next_states_b[i], axis=0))
                    max_next_q_value = np.max(next_q_values)  # Max Q-value for the next state

                    target_q_value = rewards_b[i] + gamma * max_next_q_value * (not dones_b[i])
                    q_values = model.predict(np.expand_dims(states_b[i], axis=0))
                    q_values[0][actions_b[i]] = target_q_value

                    model.fit(np.expand_dims(states_b[i], axis=0), q_values, epochs=1, verbose=0)

            state = next_state

        rewards_per_episode.append(total_reward)

    # Plot the reward progress over episodes
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Progress Over Episodes")
    plt.show()
