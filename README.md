# Federated Learning for Intrusion Detection using CNN-LSTM

This project implements a CNN-LSTM hybrid deep learning model for binary network intrusion detection, using **Federated Learning** via the Flower framework. The dataset used is a version of the **NSL-KDD** dataset.

## 💡 Objective

The goal is to simulate distributed training of a CNN-LSTM model using Federated Learning without sharing raw data between clients. Each client trains on a partitioned dataset and shares only model updates with the server.

## 📦 Dataset

- **NSL-KDD** (KDDTrain+ and KDDTest+)
- Binary classification: `normal` (0) vs `attack` (1)
- Feature engineering: Label encoding of categorical features, dropped unnecessary columns

## ⚙️ Architecture

- `Conv1D + MaxPooling1D + LSTM + Dense` layers
- Sigmoid activation for binary output
- Binary cross-entropy loss

## 🔁 Federated Learning Setup

- Framework: Flower (`flwr`)
- Clients simulated in a Jupyter Notebook environment
- Strategy: `FedAvg`
- Custom metric aggregation with per-round accuracy tracking
- Optional GPU simulation disabled for compatibility

## 📊 Output

- Validation accuracy logged per round
- Final test accuracy plotted
- Federated simulation of 10 clients over 5 rounds

## 🚀 How to Run

1. Install dependencies:
```bash
pip install flwr tensorflow scikit-learn
```

2. Ensure your data is loaded and preprocessed (`df`, `test_df`)

3. Run the notebook step-by-step:
- Define CNN-LSTM model (`create_model`)
- Partition the dataset into federated clients
- Define the Flower client and strategy
- Run `fl.simulation.start_simulation(...)`
- Plot final accuracy trends

## 📈 Example Output

- Round 1 - Weighted Accuracy: 0.899
- Round 5 - Weighted Accuracy: 0.956
- Final Test Accuracy: 0.9560


## 🙌 Acknowledgements

- Flower team for federated learning framework
- NSL-KDD dataset authors
- TensorFlow/Keras for DL framework

---
