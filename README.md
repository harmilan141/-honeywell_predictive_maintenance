# SmartPredict: Neural Network-Based Predictive Maintenance System

SmartPredict is an AI-powered predictive maintenance system designed for smart factories. It predicts potential equipment failures in advance using multi-sensor heterogeneous data streams.

## Problem Statement

Honeywell aims to design a neural network-based predictive maintenance model for smart factories. The system predicts potential equipment failures in advance using sensor streams, anomaly detection, feature fusion, and real-time inference.

## Dataset

The project uses the NASA C-MAPSS turbofan engine degradation dataset.

## Methodology

1. Data preprocessing
2. Remaining Useful Life calculation
3. Feature engineering
4. Rolling statistical feature extraction
5. Anomaly detection using Isolation Forest
6. Neural Network-based failure risk prediction
7. Real-time inference using Streamlit

## Innovation

- Hybrid anomaly detection + neural network prediction
- Multi-sensor feature fusion
- Rolling temporal sensor behavior analysis
- False alarm reduction using risk thresholds
- Real-time dashboard for maintenance decisions

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly / Matplotlib

## Output

The system predicts:

- Failure probability
- Machine status: Normal, Warning, Critical
- Maintenance priority: Low, Medium, High
- Decision support recommendation

## Run Project

```bash
streamlit run app.py