import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import urllib.request
import gzip
import os

# Download MNIST 
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",     # Google Cloud
    "https://ossci-datasets.s3.amazonaws.com/mnist/"           # AWS backup
]

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

def download_file(url, path):
    try:
        print(f"Trying: {url}")
        urllib.request.urlretrieve(url, path)
        print("Downloaded:", path)
        return True
    except Exception as e:
        print("Failed:", e)
        return False

def download_mnist():
    os.makedirs("mnist_data", exist_ok=True)
    for f in FILES:
        local_path = os.path.join("mnist_data", f)
        if os.path.exists(local_path):
            print("Already exists:", f)
            continue

        print(f"\nDownloading {f} ...")
        success = False
        for mirror in MNIST_MIRRORS:
            if download_file(mirror + f, local_path):
                success = True
                break

        if not success:
            raise Exception(f"Could not download: {f}")

def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
        _, num, rows, cols = np.frombuffer(data[:16], dtype=">u4")
        images = np.frombuffer(data[16:], dtype=np.uint8)
        return images.reshape(num, rows * cols)

def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
        _, num = np.frombuffer(data[:8], dtype=">u4")
        labels = np.frombuffer(data[8:], dtype=np.uint8)
        return labels

print("Downloading MNIST data (if missing)...")
download_mnist()

print("Loading MNIST...")

train_images = load_idx_images("mnist_data/train-images-idx3-ubyte.gz")
train_labels = load_idx_labels("mnist_data/train-labels-idx1-ubyte.gz")

test_images  = load_idx_images("mnist_data/t10k-images-idx3-ubyte.gz")
test_labels  = load_idx_labels("mnist_data/t10k-labels-idx1-ubyte.gz")

# Normalize & reshape like CSV version
X_train = (train_images / 255.).astype(np.float32).T
Y_train = train_labels.astype(int)
m_train = Y_train.size

X_test = (test_images / 255.).astype(np.float32).T
Y_test = test_labels.astype(int)
m_test = Y_test.size

print("Training samples:", m_train)
print("Test samples:", m_test)

# Building NN core

def one_hot(Y, C=10):
    oh = np.zeros((C, Y.size))
    oh[Y, np.arange(Y.size)] = 1
    return oh

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=0, keepdims=True)

def ReLU(Z): return np.maximum(0, Z)
def ReLU_deriv(Z): return (Z > 0).astype(float)

def compute_loss(A3, Y):
    m = Y.size
    oh = one_hot(Y)
    log_probs = np.log(A3 + 1e-12)
    return -np.sum(oh * log_probs) / m

def init_params():
    W1 = np.random.randn(128, 784) * np.sqrt(2.0 / (784 + 128))
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(2.0 / (128 + 64))
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(10, 64) * np.sqrt(2.0 / (64 + 10))
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def backward_prop(cache, W1, W2, W3, X, Y):
    Z1, A1, Z2, A2, Z3, A3 = cache
    m = Y.size
    Y_oh = one_hot(Y)

    dZ3 = A3 - Y_oh
    dW3 = (1/m) * dZ3.dot(A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, grads, lr):
    dW1, db1, dW2, db2, dW3, db3 = grads
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    return W1, b1, W2, b2, W3, b3

def predict_from_A(A3):
    return np.argmax(A3, axis=0)


# Training 

def train(X, Y, epochs=10, batch_size=128, lr=0.08):
    W1, b1, W2, b2, W3, b3 = init_params()
    num = Y.size
    steps = int(np.ceil(num / batch_size))

    for epoch in range(epochs):
        perm = np.random.permutation(num)
        Xs, Ys = X[:, perm], Y[perm]

        total_loss = 0

        for s in range(steps):
            a, b = s * batch_size, min((s + 1) * batch_size, num)
            Xb, Yb = Xs[:, a:b], Ys[a:b]

            cache = forward_prop(W1, b1, W2, b2, W3, b3, Xb)
            A3 = cache[5]

            total_loss += compute_loss(A3, Yb) * Yb.size

            grads = backward_prop(cache, W1, W2, W3, Xb, Yb)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, grads, lr)

        # training accuracy
        A_train = forward_prop(W1, b1, W2, b2, W3, b3, X)[5]
        pred_train = predict_from_A(A_train)
        acc_train = np.mean(pred_train == Y)

        print(f"Epoch {epoch+1}/{epochs}  Loss={total_loss/num:.4f}  Acc={acc_train:.4f}")

    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = train(X_train, Y_train, epochs=15, batch_size=128, lr=0.08)


# evaluate on test set

A3_test = forward_prop(W1, b1, W2, b2, W3, b3, X_test)[5]
y_pred_test = predict_from_A(A3_test)

num_wrong = np.sum(y_pred_test != Y_test)
accuracy_test = np.mean(y_pred_test == Y_test)

print(f"\nTest accuracy = {accuracy_test:.4f}")
print(f"Misclassified: {num_wrong}/{m_test}")


# visualize misclassified images

misclassified_idx = np.where(y_pred_test != Y_test)[0]
print(f"Total misclassified images: {len(misclassified_idx)}")

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
current_idx = [0]

def show_image(idx):
    ax.clear()
    x = X_test[:, misclassified_idx[idx]].reshape(28, 28)
    y_t = Y_test[misclassified_idx[idx]]
    y_p = y_pred_test[misclassified_idx[idx]]
    conf = A3_test[y_p, misclassified_idx[idx]] * 100

    ax.imshow(x, cmap='gray')
    ax.set_title(f"True: {y_t} | Pred: {y_p} ({conf:.2f}%)  {idx+1}/{len(misclassified_idx)}")
    ax.axis('off')
    fig.canvas.draw()

def next_image(event):
    current_idx[0] = min(current_idx[0] + 1, len(misclassified_idx) - 1)
    show_image(current_idx[0])

def prev_image(event):
    current_idx[0] = max(current_idx[0] - 1, 0)
    show_image(current_idx[0])

btn_next = Button(plt.axes([0.6, 0.05, 0.2, 0.075]), 'Next')
btn_next.on_clicked(next_image)

btn_prev = Button(plt.axes([0.3, 0.05, 0.2, 0.075]), 'Back')
btn_prev.on_clicked(prev_image)

if len(misclassified_idx) > 0:
    show_image(0)
    plt.show()
else:
    print("No misclassified images!")
