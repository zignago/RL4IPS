# Reinforcement Learning for Intrusion Prevention System (RL4IPS)

### Overview

This project implements a reinforcement learning (RL) approach to develop an Intrusion Prevention System (IPS). The RL agent is trained on network traffic data to dynamically learn how to classify network flows into benign or malicious, and take appropriate actions to either Allow, Block, or Flag network traffic. The model uses deep Q-learning to train the agent and features an experience replay memory for efficient learning.

### Dataset

The dataset used for this project is from the CICIDS2017 dataset, which contains detailed network traffic data, including both benign and various types of malicious attacks. This repository includes one of the CSV files from the dataset in the CICIDS2017/ folder for ease of use during testing.

You can find the full dataset on Kaggle here: [Network Intrusion Dataset (CICIDS2017)](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

### Project Structure

The code is organized into different modules to ensure maintainability and separation of concerns. Below is the structure of the repository:

```
/core
    ├── data_loader.py       # Handles data loading and preprocessing
    ├── model.py             # Defines the Q-learning model (neural network)
    ├── epsilon_decay.py     # Implements epsilon decay and action selection
    ├── environment.py       # Manages state, action, and reward interactions with the environment
    └── training.py          # Contains the training loop and experience replay functionality
CICIDS2017/
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv  # Sample CSV file for network traffic data
rl4ips.py                    # Main script to train the RL agent
```

### Files Description

`core/data_loader.py`: Contains functions to load and preprocess the network traffic data, including feature extraction and label formatting.

`core/model.py`: Defines the deep Q-learning network that will be used for training the RL agent.

`core/epsilon_decay.py`: Manages exploration-exploitation trade-offs using epsilon-greedy action selection and decay functions.

`core/environment.py`: Defines the environment interactions, where the agent can take actions and receive feedback in the form of rewards.

`core/training.py`: Implements the training loop with experience replay memory to train the agent effectively.

`rl4ips.py`: The main script that integrates all the modules, loads the data, trains the agent, and visualizes the results.

### Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- You can install the required libraries using pip:

pip install -r requirements.txt

### Running the Project

``` python rl4ips.py ```

### License
This project is licensed under the MIT License.
