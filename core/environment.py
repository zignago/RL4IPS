import numpy as np

def get_initial_state(data, index=None):
    if index is None:
        index = np.random.randint(0, len(data))
    state, label = data[index]
    return state, label

def execute_action(action, current_state, step, max_steps_per_episode, data):
    next_state, label = get_initial_state(data)  # Get next state and its label (benign or malicious)
    
    # Define reward structure
    if action == 0:  # Allow
        reward = -3 if label == 1 else 2  # Penalty for allowing malicious, reward for benign
    elif action == 1:  # Block
        reward = 3 if label == 1 else -1  # Reward for blocking malicious, penalty for false positive
    else:  # Flag
        reward = 0.5 if label == 1 else 0  # Small reward for flagging malicious, neutral for benign

    done = (step >= max_steps_per_episode) or (np.random.rand() > 0.98)  # Properly end episodes
    return next_state, reward, done
