import numpy as np
import matplotlib.pyplot as plt


# ===========================
# 0. Generate synthetic data
# ===========================

def generate_sine_series(n_points=2000, noise_std=0.1, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 40 * np.pi, n_points)  # long time range
    x = np.sin(t) + 0.3 * np.sin(3 * t) + noise_std * rng.randn(n_points)
    return x  # shape: [n_points]


def make_windows(series, seq_len=20):
    """
    Turn 1D series into (X, y) pairs using sliding windows.

    X[i] = [x_t, ..., x_{t+seq_len-1}]
    y[i] = x_{t+seq_len}
    """
    X = []
    y = []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X)  # [N, seq_len]
    y = np.array(y)  # [N]
    # reshape to [N, seq_len, 1] for RNN input
    X = X[..., None]
    y = y[..., None]
    return X, y


# =========================================
# 1. Vanilla RNN implemented with NumPy
# =========================================

class SimpleRNN:
    def __init__(self, input_size=1, hidden_size=32, learning_rate=0.001, seed=0):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate

        # Initialize parameters
        # Wx: [H, D], Wh: [H, H], bh: [H, 1]
        self.Wx = rng.randn(hidden_size, input_size) * 0.1
        self.Wh = rng.randn(hidden_size, hidden_size) * 0.1
        self.bh = np.zeros((hidden_size, 1))

        # Output layer: Wy: [1, H], by: [1, 1]
        self.Wy = rng.randn(1, hidden_size) * 0.1
        self.by = np.zeros((1, 1))

    def forward(self, x_seq):
        """
        x_seq: [T, D]  (D = 1 here)
        Returns:
            y_pred: [1, 1]
            cache: dict for BPTT
        """
        T, D = x_seq.shape
        H = self.hidden_size

        # Store all hidden states for backprop
        h_list = []
        # Initial hidden state h0 = 0
        h_prev = np.zeros((H, 1))

        for t in range(T):
            x_t = x_seq[t].reshape(D, 1)  # [D, 1]
            # TODO: compute h_t = tanh(Wx x_t + Wh h_prev + bh)
            a_t = self.Wx @ x_t + self.Wh @ h_prev + self.bh
            h_t = np.tanh(a_t)

            h_list.append(h_t)
            h_prev = h_t

        # Output layer uses last hidden state h_T
        h_T = h_list[-1]  # [H, 1]
        # TODO: compute y_pred = Wy h_T + by
        y_pred = self.Wy @ h_T + self.by  # [1, 1]

        cache = {
            "x_seq": x_seq,
            "h_list": h_list
        }
        return y_pred, cache

    def backward(self, y_pred, y_true, cache):
        """
        Compute gradients for one sequence using BPTT.
        y_pred, y_true: [1, 1]
        Returns:
            grads: dict of dWx, dWh, dbh, dWy, dby
        """
        x_seq = cache["x_seq"]
        h_list = cache["h_list"]

        T, D = x_seq.shape
        H = self.hidden_size

        # Initialize gradients with zeros
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        # --------- Gradients at output layer ----------
        # Loss: L = 0.5 * (y_pred - y_true)^2
        # TODO: dL/dy_pred
        dL_dy = (y_pred - y_true)  # [1, 1]

        h_T = h_list[-1]  # [H, 1]
        # TODO: dWy, dby, and gradient w.r.t. h_T
        dWy += dL_dy @ h_T.T      # [1, H]
        dby += dL_dy              # [1, 1]
        dh_next = self.Wy.T @ dL_dy  # [H, 1], gradient wrt h_T

        # --------- BPTT through time ----------
        # Iterate backwards over time steps
        for t in reversed(range(T)):
            h_t = h_list[t]
            x_t = x_seq[t].reshape(D, 1)
            h_prev = h_list[t-1] if t > 0 else np.zeros_like(h_t)

            # dh includes gradient from future time steps
            dh = dh_next

            # Derivative of tanh: d(tanh)/da = 1 - h^2
            # TODO: compute delta_t = dh * (1 - h_t^2)
            delta_t = dh * (1.0 - h_t ** 2)  # [H, 1]

            # Gradients for parameters
            dWx += delta_t @ x_t.T      # [H, D]
            dWh += delta_t @ h_prev.T   # [H, H]
            dbh += delta_t             # [H, 1]

            # Propagate gradient to previous hidden state
            dh_next = self.Wh.T @ delta_t  # [H, 1]

        grads = {
            "dWx": dWx,
            "dWh": dWh,
            "dbh": dbh,
            "dWy": dWy,
            "dby": dby
        }
        return grads

    def update_params(self, grads):
        # SGD update
        self.Wx -= self.lr * grads["dWx"]
        self.Wh -= self.lr * grads["dWh"]
        self.bh -= self.lr * grads["dbh"]
        self.Wy -= self.lr * grads["dWy"]
        self.by -= self.lr * grads["dby"]


# =========================================
# 2. Training utilities
# =========================================

def train_rnn(model, X_train, y_train, n_epochs=20):
    """
    X_train: [N, T, 1]
    y_train: [N, 1]
    """
    N, T, D = X_train.shape
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data indices
        indices = np.arange(N)
        np.random.shuffle(indices)

        total_loss = 0.0
        for idx in indices:
            x_seq = X_train[idx]        # [T, 1]
            y_true = y_train[idx].reshape(1, 1)  # [1, 1]

            # Forward
            y_pred, cache = model.forward(x_seq)

            # Compute loss
            loss = 0.5 * np.sum((y_pred - y_true) ** 2)
            total_loss += loss

            # Backward
            grads = model.backward(y_pred, y_true, cache)

            # Update
            model.update_params(grads)

        avg_loss = total_loss / N
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, loss = {avg_loss:.6f}")

    return losses


def evaluate_rnn(model, X, y):
    N, T, D = X.shape
    total_loss = 0.0
    preds = []

    for i in range(N):
        x_seq = X[i]
        y_true = y[i].reshape(1, 1)
        y_pred, _ = model.forward(x_seq)
        loss = 0.5 * np.sum((y_pred - y_true) ** 2)
        total_loss += loss
        preds.append(y_pred.item())

    mse = (2 * total_loss) / N  # because loss has 0.5 factor
    return mse, np.array(preds)


# =========================================
# 3. Main script
# =========================================

if __name__ == "__main__":
    # 0. Prepare data
    series = generate_sine_series(n_points=2500, noise_std=0.1, seed=42)
    seq_len = 20
    X, y = make_windows(series, seq_len=seq_len)

    # Train/test split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # 1. Create model
    model = SimpleRNN(input_size=1, hidden_size=32, learning_rate=0.001, seed=0)

    # 2. Train
    losses = train_rnn(model, X_train, y_train, n_epochs=20)

    # 3. Evaluate
    mse, y_pred_test = evaluate_rnn(model, X_test, y_test)
    print("Test MSE:", mse)

    # 4. Plot a small segment of predictions vs ground truth
    plt.figure(figsize=(10, 4))

    # Choose a slice in the test set
    n_show = 100
    # corresponding time indices in original series (approx)
    test_start_idx = split + seq_len
    time_idx = np.arange(test_start_idx, test_start_idx + n_show)

    plt.plot(time_idx, series[test_start_idx:test_start_idx + n_show],
             label="True series")
    plt.plot(time_idx, y_pred_test[:n_show],
             label="RNN prediction (1-step ahead)")
    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Vanilla RNN – Sine Wave One-step Prediction")
    plt.tight_layout()
    plt.show()
