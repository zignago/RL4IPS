from core.data_loader import load_data
from core.model import create_q_model
from core.training import train_and_visualize

# Constants
EPISODES = 1000
BATCH_SIZE = 64
MAX_STEPS_PER_EPISODE = 30
GAMMA = 0.95
EPSILON_START = 1.0
REPLAY_MEMORY_SIZE = 5000
LEARNING_RATE = 0.0001

# Load data
data = load_data('CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv')

# Create model
model = create_q_model(state_size=10, num_actions=3, learning_rate=LEARNING_RATE)

# Train and visualize results
train_and_visualize(model, data, EPISODES, MAX_STEPS_PER_EPISODE, BATCH_SIZE, GAMMA, EPSILON_START, REPLAY_MEMORY_SIZE)
