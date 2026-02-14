# Political Sentiment Shift Prediction

A comprehensive data science project for analyzing and predicting political sentiment shifts using Neural Networks (LSTM) and Time-Series Forecasting.

## Project Structure
The project is organized as a modular Python package:

```
.
├── src/
│   ├── data.py           # Data loading and processing
│   ├── preprocessing.py  # Cleaning and tokenization
│   ├── model.py          # LSTM Neural Network definition
│   ├── visualization.py  # Plotting utilities
│   ├── config.py         # Configuration and constants
│   └── utils.py          # Utility functions
├── main.py               # Main entry point
├── run.sh                # Automation script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Features
-   **Sentiment Analysis**: Uses DistilBERT transformers to analyze text sentiment.
-   **Time-Series Aggregation**: Converts article sentiment into daily time-series data.
-   **Shift Prediction**: Uses LSTM to predict future sentiment shifts (Positive/Negative/Stable).
-   **Visualization**: Generates comprehensive plots for analysis and training history.

## Setup and Usage

### Prerequisites
-   Python 3.8+
-   Git

### Quick Start
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/adnanquraishee/PoliticalSentimentShift.git
    cd PoliticalSentimentShift
    ```

2.  **Run the automated script**:
    The `run.sh` script handles virtual environment creation, dependency installation, and execution.
    ```bash
    ./run.sh
    ```

### Manual Setup
If you prefer to run it manually:

1.  Create virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the pipeline:
    ```bash
    python main.py
    ```

## Outputs
Artifacts such as models and plots are saved in the root directory and `model_artifacts/`.
