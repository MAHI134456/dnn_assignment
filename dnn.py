
import numpy as np

np.random.seed(42)


# --------------------------------------------------------------------------
# Activation functions and their derivatives
# --------------------------------------------------------------------------
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(a):  # a = sigmoid(x) already computed
    return a * (1 - a)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(a):  # a = tanh(x)
    return 1 - a ** 2


def relu(x):
    return np.maximum(0, x)


def relu_deriv(a):
    return (a > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)


def prelu_deriv_x(x, alpha):
    return np.where(x > 0, 1.0, alpha)


ACTS = {
    "sigmoid": (sigmoid, sigmoid_deriv, "post"),   # deriv uses post-activation value
    "tanh": (tanh, tanh_deriv, "post"),
    "relu": (relu, relu_deriv, "post"),
    "leaky_relu": (leaky_relu, leaky_relu_deriv, "pre"),   # deriv uses pre-activation (z)
}


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y, num_classes):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh


class SimpleDNN:
    """Fully connected network: input -> [hidden layers with `activation`] -> softmax output."""

    def __init__(self, layer_sizes, activation="relu", lr=0.05, prelu_init=0.25, seed=0):
        """
        layer_sizes: e.g. [64, 32, 32, 32, 10]  (first = input dim, last = num classes)
        activation: 'sigmoid' | 'tanh' | 'relu' | 'leaky_relu' | 'prelu'
        """
        rng = np.random.RandomState(seed)
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.lr = lr
        self.n_layers = len(layer_sizes) - 1  # number of weight matrices (last is output/softmax)
        self.W, self.b = [], []
        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            # Same simple initialization scale for every activation on purpose:
            # this keeps the comparison in experiment 1.1 fair (architecture &
            # init identical, only the activation function changes).
            W = rng.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)
            self.W.append(W)
            self.b.append(np.zeros((1, fan_out)))

        # learnable per-layer alpha for PReLU (one scalar per hidden layer)
        self.alpha = [np.array(prelu_init) for _ in range(self.n_layers - 1)]

    def _hidden_act(self, z, layer_idx):
        if self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "tanh":
            return tanh(z)
        elif self.activation == "relu":
            return relu(z)
        elif self.activation == "leaky_relu":
            return leaky_relu(z)
        elif self.activation == "prelu":
            return prelu(z, self.alpha[layer_idx])
        else:
            raise ValueError(self.activation)

    def _hidden_act_deriv(self, z, a, layer_idx):
        if self.activation == "sigmoid":
            return sigmoid_deriv(a)
        elif self.activation == "tanh":
            return tanh_deriv(a)
        elif self.activation == "relu":
            return relu_deriv(a)
        elif self.activation == "leaky_relu":
            return leaky_relu_deriv(z)
        elif self.activation == "prelu":
            return prelu_deriv_x(z, self.alpha[layer_idx])
        else:
            raise ValueError(self.activation)

    def forward(self, X):
        Zs, As = [], [X]
        a = X
        for i in range(self.n_layers - 1):  # hidden layers
            z = a @ self.W[i] + self.b[i]
            a = self._hidden_act(z, i)
            Zs.append(z)
            As.append(a)
        # output layer (softmax)
        z_out = a @ self.W[-1] + self.b[-1]
        out = softmax(z_out)
        Zs.append(z_out)
        As.append(out)
        return Zs, As

    def backward(self, Zs, As, y_onehot):
        m = y_onehot.shape[0]
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers
        grad_alpha = [None] * (self.n_layers - 1)

        # output layer: softmax + cross-entropy -> simple gradient
        dz = (As[-1] - y_onehot) / m
        grads_W[-1] = As[-2].T @ dz
        grads_b[-1] = np.sum(dz, axis=0, keepdims=True)

        layer_grad_norms = [None] * self.n_layers
        layer_grad_norms[-1] = np.mean(np.abs(grads_W[-1]))

        da = dz @ self.W[-1].T
        for i in reversed(range(self.n_layers - 1)):
            z_i, a_i = Zs[i], As[i + 1]
            deriv = self._hidden_act_deriv(z_i, a_i, i)
            dz_i = da * deriv

            if self.activation == "prelu":
                neg_mask = (z_i < 0)
                grad_alpha[i] = np.sum(da * z_i * neg_mask)

            grads_W[i] = As[i].T @ dz_i
            grads_b[i] = np.sum(dz_i, axis=0, keepdims=True)
            layer_grad_norms[i] = np.mean(np.abs(grads_W[i]))

            da = dz_i @ self.W[i].T

        return grads_W, grads_b, grad_alpha, layer_grad_norms

    def step(self, grads_W, grads_b, grad_alpha):
        for i in range(self.n_layers):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]
        if self.activation == "prelu":
            for i in range(self.n_layers - 1):
                self.alpha[i] -= self.lr * grad_alpha[i]
                self.alpha[i] = np.clip(self.alpha[i], -1.0, 1.0)

    def predict(self, X):
        _, As = self.forward(X)
        return np.argmax(As[-1], axis=1)

    def loss(self, X, y_onehot):
        _, As = self.forward(X)
        p = np.clip(As[-1], 1e-12, 1.0)
        return -np.mean(np.sum(y_onehot * np.log(p), axis=1))

    def dead_neuron_fraction(self, X):
        """Fraction of hidden units that output exactly 0 for the whole batch
        (only meaningful for ReLU-family activations)."""
        Zs, As = self.forward(X)
        dead, total = 0, 0
        for i in range(self.n_layers - 1):
            a = As[i + 1]
            dead += np.sum(np.all(a == 0, axis=0))
            total += a.shape[1]
        return dead / total

    def train(self, X, y, X_val, y_val, num_classes, epochs=60, batch_size=32, track_grads=True):
        y_oh = one_hot(y, num_classes)
        y_val_oh = one_hot(y_val, num_classes)
        n = X.shape[0]
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                   "grad_norms_per_epoch": []}  # grad_norms_per_epoch[e] = list per layer

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_sh, y_oh_sh = X[perm], y_oh[perm]
            epoch_grad_norms = None
            n_batches = 0
            for start in range(0, n, batch_size):
                xb = X_sh[start:start + batch_size]
                yb = y_oh_sh[start:start + batch_size]
                Zs, As = self.forward(xb)
                grads_W, grads_b, grad_alpha, layer_grad_norms = self.backward(Zs, As, yb)
                self.step(grads_W, grads_b, grad_alpha)
                if track_grads:
                    if epoch_grad_norms is None:
                        epoch_grad_norms = np.array(layer_grad_norms, dtype=float)
                    else:
                        epoch_grad_norms += np.array(layer_grad_norms, dtype=float)
                    n_batches += 1

            if track_grads:
                history["grad_norms_per_epoch"].append((epoch_grad_norms / n_batches).tolist())

            train_loss = self.loss(X, y_oh)
            val_loss = self.loss(X_val, y_val_oh)
            train_acc = np.mean(self.predict(X) == y)
            val_acc = np.mean(self.predict(X_val) == y_val)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

        return history
