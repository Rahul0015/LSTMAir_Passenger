from utils import load_data, decompose_series, prepare_sequences
from models.LSTM_model import build_and_train_lstm_pytorch, load_trained_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import os
import numpy as np
# Load and preprocess data
df = load_data("data/AirPassengers.csv")
decompose_series(df)
X, y, scaler = prepare_sequences(df, time_steps=10)

# Train/test split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model_path = "models/lstm_air_passengers.pth"

# Load model if exists, else train and save
if os.path.exists(model_path):
    print("Loading trained model...")
    model = load_trained_model(model_path)
else:
    print("Training model...")
    model = build_and_train_lstm_pytorch(X_train, y_train, time_steps=10, epochs=50, batch_size=16, learning_rate=0.001, save_path=model_path)

# Plot and evaluate
def plot_predictions(model, X_test, y_test, scaler):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()

    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label='Actual', color='blue')
    plt.plot(predictions_inv, label='Predicted', color='orange')
    plt.title('LSTM Forecast: Actual vs. Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Passengers')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    mse = mean_squared_error(y_test_inv, predictions_inv)
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_inv, predictions_inv)
    print("R² Score:", r2)

def forecast_future(model, X_input, scaler, steps=12):
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Input Validation
    print("\n=== Input Validation ===")
    print(f"Raw X_input shape: {X_input.shape}")
    print(f"Last 3 X_input values:\n{X_input[-1,-3:,0]}")
    
    last_sequence = X_input[-1:].reshape(1, -1, 1)  # Force (1, timesteps, 1)
    print(f"\nLast sequence shape: {last_sequence.shape}")
    print(f"Last sequence values:\n{last_sequence.squeeze()}")
    
    # 2. Tensor Conversion
    input_seq = torch.tensor(last_sequence, dtype=torch.float32).to(device)
    print(f"\nInitial tensor values:\n{input_seq.cpu().numpy().squeeze()}")
    
    preds = []
    history = []  # To track full sequence evolution

    with torch.no_grad():
        for i in range(steps):
            # 3. Prediction Step
            pred = model(input_seq)
            pred_value = pred.item()
            preds.append(pred_value)
            
            print(f"\n=== Step {i} ===")
            print(f"Input shape: {input_seq.shape}")
            print(f"Input values:\n{input_seq.cpu().numpy().squeeze()}")
            print(f"Predicted (scaled): {pred_value:.4f}")
            
            # 4. Sequence Update
            pred_reshaped = pred.reshape(1, 1, 1)
            input_seq = torch.cat((input_seq[:, 1:, :], pred_reshaped), dim=1)
            history.append(input_seq.clone().cpu().numpy())
            
            print(f"Updated sequence:\n{input_seq.cpu().numpy().squeeze()}")

    # 5. Result Processing
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    
    print("\n=== Final Results ===")
    print("All predictions (scaled):", preds.squeeze())
    print("All predictions (raw):", preds_inv.squeeze())
    
    # 6. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(preds_inv, marker='o', linestyle='--', color='red')
    plt.title(f"{steps}-Month Forecast\nFirst predicted value: {preds_inv[0][0]:.1f}")
    plt.grid(True)
    plt.show()
    
    return preds_inv

plot_predictions(model, X_test, y_test, scaler)
forecast_future(model, X_test, scaler, steps=12)
