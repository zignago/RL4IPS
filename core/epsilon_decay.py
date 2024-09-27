import numpy as np

def epsilon_decay(episode, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.995):
    return max(min_epsilon, initial_epsilon * (decay_rate ** episode))

def choose_action(state, epsilon, model, num_actions):
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)  # Random action (exploration)
    else:
        q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)  # Predict action values
        return np.argmax(q_values)  # Choose action with highest Q-value
