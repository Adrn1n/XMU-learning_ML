import numpy as np
import struct
import gzip
import urllib.request


# =========================
#       LeNet 模型
# =========================
class LeNet:
    def __init__(self):
        # conv1: 输入 1x28x28 -> 输出 6x28x28 （padding=2, kernel=5）
        self.conv1_weights = np.random.randn(6, 1, 5, 5) * 0.1
        self.conv1_bias = np.zeros(6)

        # conv2: 输入 6x14x14 -> 输出 16x10x10 （无 padding, kernel=5）
        self.conv2_weights = np.random.randn(16, 6, 5, 5) * 0.1
        self.conv2_bias = np.zeros(16)

        # 全连接层，采用 (in_dim, out_dim) 形状
        self.fc1_weights = np.random.randn(16 * 5 * 5, 120) * 0.1
        self.fc1_bias = np.zeros(120)

        self.fc2_weights = np.random.randn(120, 84) * 0.1
        self.fc2_bias = np.zeros(84)

        self.fc3_weights = np.random.randn(84, 10) * 0.1
        self.fc3_bias = np.zeros(10)

    # ---------- 基础函数 ----------

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, dout, x):
        """ReLU 对输入 x 的反向传播"""
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx

    def softmax(self, x):
        """按行（每个样本）做 softmax"""
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    # ---------- 卷积与其反向 ----------

    def convolve(self, x, W, b, stride=1, padding=0):
        """
        x: (B, C_in, H, W)
        W: (C_out, C_in, kH, kW)
        b: (C_out,)
        """
        batch_size, in_c, H, W_in = x.shape
        out_c, _, kH, kW = W.shape

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W_in + 2 * padding - kW) // stride + 1

        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
        out = np.zeros((batch_size, out_c, H_out, W_out))

        for b_idx in range(batch_size):
            for oc in range(out_c):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = x_padded[
                            b_idx, :, h_start : h_start + kH, w_start : w_start + kW
                        ]
                        out[b_idx, oc, i, j] = np.sum(region * W[oc]) + b[oc]

        cache = (x, W, b, stride, padding)
        return out, cache

    def conv_backward(self, dout, cache):
        x, W, b, stride, padding = cache
        batch_size, in_c, H, W_in = x.shape
        out_c, _, kH, kW = W.shape
        _, _, H_out, W_out = dout.shape

        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(W)
        db = np.zeros_like(b)

        for b_idx in range(batch_size):
            for oc in range(out_c):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        region = x_padded[
                            b_idx, :, h_start : h_start + kH, w_start : w_start + kW
                        ]
                        db[oc] += dout[b_idx, oc, i, j]
                        dW[oc] += dout[b_idx, oc, i, j] * region
                        dx_padded[
                            b_idx,
                            :,
                            h_start : h_start + kH,
                            w_start : w_start + kW,
                        ] += (
                            dout[b_idx, oc, i, j] * W[oc]
                        )

        if padding > 0:
            dx = dx_padded[:, :, padding:-padding, padding:-padding]
        else:
            dx = dx_padded

        return dx, dW, db

    # ---------- 最大池化与反向 ----------

    def max_pool(self, x, size=2, stride=2):
        """
        x: (B, C, H, W)
        """
        batch_size, channels, H, W_in = x.shape
        H_out = (H - size) // stride + 1
        W_out = (W_in - size) // stride + 1
        out = np.zeros((batch_size, channels, H_out, W_out))

        for b_idx in range(batch_size):
            for c in range(channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x[
                            b_idx,
                            c,
                            h_start : h_start + size,
                            w_start : w_start + size,
                        ]
                        out[b_idx, c, i, j] = np.max(window)

        cache = (x, size, stride, out)
        return out, cache

    def max_pool_backward(self, dout, cache):
        x, size, stride, out = cache
        batch_size, channels, H, W_in = x.shape
        _, _, H_out, W_out = dout.shape

        dx = np.zeros_like(x)
        for b_idx in range(batch_size):
            for c in range(channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        window = x[
                            b_idx,
                            c,
                            h_start : h_start + size,
                            w_start : w_start + size,
                        ]
                        m = np.max(window)
                        # 把梯度传给窗口中最大的元素
                        for p in range(size):
                            for q in range(size):
                                if window[p, q] == m:
                                    dx[b_idx, c, h_start + p, w_start + q] += dout[
                                        b_idx, c, i, j
                                    ]
        return dx

    # ---------- 前向传播 ----------

    def forward(self, x, training=False):
        # conv1 + ReLU
        self.c1, self.c1_cache = self.convolve(
            x, self.conv1_weights, self.conv1_bias, stride=1, padding=2
        )
        self.a1 = self.relu(self.c1)

        # pool1
        self.p1, self.p1_cache = self.max_pool(self.a1, size=2, stride=2)

        # conv2 + ReLU
        self.c2, self.c2_cache = self.convolve(
            self.p1, self.conv2_weights, self.conv2_bias, stride=1, padding=0
        )
        self.a2 = self.relu(self.c2)

        # pool2
        self.p2, self.p2_cache = self.max_pool(self.a2, size=2, stride=2)

        # 展平
        self.flat = self.p2.reshape(self.p2.shape[0], -1)

        # fc1 + ReLU
        self.z1 = self.flat @ self.fc1_weights + self.fc1_bias
        self.h1 = self.relu(self.z1)

        # fc2 + ReLU
        self.z2 = self.h1 @ self.fc2_weights + self.fc2_bias
        self.h2 = self.relu(self.z2)

        # fc3 + softmax
        self.scores = self.h2 @ self.fc3_weights + self.fc3_bias
        probs = self.softmax(self.scores)

        return probs

    # ---------- 反向传播（整网） ----------

    def backward(self, dprobs, lr=0.01):
        """
        dprobs: softmax 输出的梯度（通常是 (p - one_hot)/batch_size）
        """
        # fc3
        d_scores = dprobs
        dW3 = self.h2.T @ d_scores  # (84, B) @ (B, 10) -> (84, 10)
        db3 = d_scores.sum(axis=0)
        dh2 = d_scores @ self.fc3_weights.T

        # fc2
        dz2 = self.relu_backward(dh2, self.z2)
        dW2 = self.h1.T @ dz2  # (120, B) @ (B, 84)
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.fc2_weights.T

        # fc1
        dz1 = self.relu_backward(dh1, self.z1)
        dW1 = self.flat.T @ dz1  # (400, B) @ (B, 120)
        db1 = dz1.sum(axis=0)
        dflat = dz1 @ self.fc1_weights.T  # (B, 400)
        dp2 = dflat.reshape(self.p2.shape)

        # pool2
        da2 = self.max_pool_backward(dp2, self.p2_cache)

        # conv2
        dc2 = self.relu_backward(da2, self.c2)
        dp1, dWc2, dbc2 = self.conv_backward(dc2, self.c2_cache)

        # pool1
        da1 = self.max_pool_backward(dp1, self.p1_cache)

        # conv1
        dc1 = self.relu_backward(da1, self.c1)
        _, dWc1, dbc1 = self.conv_backward(dc1, self.c1_cache)

        # 参数更新（SGD）
        self.fc3_weights -= lr * dW3
        self.fc3_bias -= lr * db3

        self.fc2_weights -= lr * dW2
        self.fc2_bias -= lr * db2

        self.fc1_weights -= lr * dW1
        self.fc1_bias -= lr * db1

        self.conv2_weights -= lr * dWc2
        self.conv2_bias -= lr * dbc2

        self.conv1_weights -= lr * dWc1
        self.conv1_bias -= lr * dbc1


# =========================
#     MNIST 数据加载
# =========================

url_base = "http://yann.lecun.com/exdb/mnist/"
files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    """如果本地没有这四个文件，可以调用此函数下载（可能会比较慢/连不上）"""
    for key, value in files.items():
        print("Downloading", value)
        urllib.request.urlretrieve(url_base + value, value)


def load_images(filename):
    with gzip.open(filename, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, 1, rows, cols).astype(np.float32) / 255.0
        return data


def load_labels(filename):
    with gzip.open(filename, "rb") as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# =========================
#     训练与测试函数
# =========================


def train_model(lenet, train_images, train_labels, epochs=1, batch_size=64, lr=0.01):
    num_samples = train_images.shape[0]
    for epoch in range(epochs):
        # 打乱
        perm = np.random.permutation(num_samples)
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        running_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            inputs = train_images[i : i + batch_size]
            labels = train_labels[i : i + batch_size]
            batch_size_actual = inputs.shape[0]

            # 前向
            probs = lenet.forward(inputs)

            # 交叉熵损失
            log_probs = -np.log(probs[np.arange(batch_size_actual), labels] + 1e-8)
            loss = log_probs.mean()
            running_loss += loss
            num_batches += 1

            # softmax + CE 的梯度
            dprobs = probs.copy()
            dprobs[np.arange(batch_size_actual), labels] -= 1
            dprobs /= batch_size_actual

            # 反向传播 + 更新
            lenet.backward(dprobs, lr=lr)

        print(f"Epoch {epoch + 1}/{epochs}, " f"Loss: {running_loss / num_batches:.4f}")


def test_model(lenet, test_images, test_labels):
    probs = lenet.forward(test_images)
    predictions = np.argmax(probs, axis=1)
    accuracy = (predictions == test_labels).mean()
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")


# =========================
#           主程序
# =========================
if __name__ == "__main__":
    # 如果你本地还没有数据，可以先运行一次：
    # download_mnist()

    train_images = load_images(files["train_images"])
    train_labels = load_labels(files["train_labels"])
    test_images = load_images(files["test_images"])
    test_labels = load_labels(files["test_labels"])

    # ===== 只用前 10000 个样本做训练（速度立刻快 6 倍左右）=====
    N = 10000
    train_images = train_images[:N]
    train_labels = train_labels[:N]

    lenet = LeNet()

    # 为了不太慢，先跑 1 个 epoch 试试，确认代码无误
    train_model(lenet, train_images, train_labels, epochs=1, batch_size=64, lr=0.01)
    test_model(lenet, test_images, test_labels)
