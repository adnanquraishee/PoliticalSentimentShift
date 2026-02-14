import torch

# Data Configuration
FORECAST_HORIZON = 3
SENTIMENT_CHANGE_THRESHOLD = 0.05
SEQUENCE_LENGTH = 5

# Model Configuration
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 50
PATIENCE = 10
WEIGHT_DECAY = 1e-5

# System Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
DATA_FILE = 'politics18.csv'  # Not used in sample gen but good to have
