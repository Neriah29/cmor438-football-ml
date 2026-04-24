"""
Multi-Layer Perceptron (MLP) Neural Network
=============================================
From-scratch NumPy implementation for CMOR 438 / INDE 577.
Author: Neriah29
"""

import numpy as np


class MLP:
    """
    Multi-Layer Perceptron for binary classification.

    Architecture: Input → [Hidden layers with ReLU] → Output with Sigmoid
    Training:     Batch Gradient Descent with Backpropagation
    Loss:         Binary Cross-Entropy

    Parameters
    ----------
    hidden_layer_sizes : tuple of int
        Number of neurons in each hidden layer.
        e.g. (64, 32) means two hidden layers — 64 neurons then 32 neurons.
    learning_rate : float
        Step size for gradient descent. Default 0.01.
    n_epochs : int
        Number of full passes over the training data. Default 1000.
    random_state : int or None
        Seed for reproducible weight initialization.

    Attributes
    ----------
    weights_ : list of np.ndarray
        Learned weight matrices for each layer.
    biases_ : list of np.ndarray
        Learned bias vectors for each layer.
    loss_history_ : list of float
        Cross-entropy loss recorded after every epoch.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.weights_ = None
        self.biases_ = None
        self.loss_history_ = []

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def _relu(self, z: np.ndarray) -> np.ndarray:
        """
        ReLU activation: max(0, z)

        Used in hidden layers. Passes positive values unchanged,
        blocks negative values (outputs 0).
        Simple but very effective — avoids the vanishing gradient
        problem that sigmoid suffers from in deep networks.
        """
        return np.maximum(0, z)

    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU — used during backpropagation.

        ReLU's derivative is:
          1 if z > 0  (active neuron — error passes through)
          0 if z <= 0 (inactive neuron — error is blocked)

        This is the "gate" in backpropagation.
        """
        return (z > 0).astype(float)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation: 1 / (1 + e^{-z})

        Used only at the output layer to produce a probability in (0, 1).
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _cross_entropy(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        p = np.clip(y_prob, 1e-12, 1 - 1e-12)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self, n_features: int, rng: np.random.Generator):
        """
        Initialize weights using He initialization.

        He initialization sets weights from a normal distribution scaled
        by sqrt(2 / n_inputs_to_layer). This keeps the variance of
        activations stable as signals pass through many layers —
        preventing them from exploding or vanishing.

        Much better than pure random initialization for deep networks.
        """
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # He initialization scale factor
            scale = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, scale, size=(fan_in, layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights_.append(W)
            self.biases_.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray):
        """
        Pass data through all layers from input to output.

        Stores intermediate values (pre-activations z and
        post-activations a) because backpropagation needs them.

        Returns
        -------
        activations : list of np.ndarray
            Output of each layer (including input). activations[-1] is
            the final probability predictions.
        pre_activations : list of np.ndarray
            Weighted sums z before activation, for each layer.
            Needed by backprop to compute ReLU derivatives.
        """
        activations = [X]       # activations[0] = raw input
        pre_activations = []    # z values before activation

        current = X
        n_layers = len(self.weights_)

        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = current @ W + b
            pre_activations.append(z)

            if i < n_layers - 1:
                # Hidden layers: ReLU
                a = self._relu(z)
            else:
                # Output layer: Sigmoid
                a = self._sigmoid(z)

            activations.append(a)
            current = a

        return activations, pre_activations

    # ------------------------------------------------------------------
    # Backward pass (backpropagation)
    # ------------------------------------------------------------------

    def _backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: list,
        pre_activations: list,
    ):
        """
        Compute gradients for all weights and biases via backpropagation.

        Works backwards from the output layer to the first hidden layer.
        At each layer:
          1. Receive error signal (delta) from the layer above
          2. Compute gradients for this layer's weights and biases
          3. Pass error signal backwards through this layer's weights
             and activation derivative

        Returns
        -------
        grad_W : list of np.ndarray  — gradients for each weight matrix
        grad_b : list of np.ndarray  — gradients for each bias vector
        """
        n_samples = X.shape[0]
        n_layers = len(self.weights_)
        grad_W = [None] * n_layers
        grad_b = [None] * n_layers

        # Output layer delta:
        # For sigmoid + cross-entropy, delta = prediction - true label
        # (same elegant result as Logistic Regression)
        delta = activations[-1] - y.reshape(-1, 1)   # shape (n, 1)

        # Backpropagate through each layer (output → first hidden)
        for i in reversed(range(n_layers)):
            # Gradient for this layer's weights and bias
            grad_W[i] = (activations[i].T @ delta) / n_samples
            grad_b[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                # Pass error backwards through this layer's weights
                # then through the previous layer's ReLU derivative
                delta = (delta @ self.weights_[i].T) * self._relu_derivative(pre_activations[i - 1])

        return grad_W, grad_b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        """
        Train the network via batch gradient descent + backpropagation.

        Each epoch:
          1. Forward pass — compute predictions and store activations
          2. Compute loss
          3. Backward pass — compute gradients for all weights
          4. Update all weights simultaneously

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary {0, 1}

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)
        self._init_weights(X.shape[1], rng)
        self.loss_history_ = []

        for _ in range(self.n_epochs):
            # Forward
            activations, pre_activations = self._forward(X)

            # Loss
            y_prob = activations[-1].ravel()
            self.loss_history_.append(self._cross_entropy(y, y_prob))

            # Backward
            grad_W, grad_b = self._backward(X, y, activations, pre_activations)

            # Update all weights
            for i in range(len(self.weights_)):
                self.weights_[i] -= self.learning_rate * grad_W[i]
                self.biases_[i]  -= self.learning_rate * grad_b[i]

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities for class 1.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in (0, 1)
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        activations, _ = self._forward(X)
        return activations[-1].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return hard binary labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        threshold : float

        Returns
        -------
        np.ndarray, shape (n_samples,) — values in {0, 1}
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))
