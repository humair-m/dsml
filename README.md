# dsml

A hands-on collection of **Data Science** and **Machine Learning** implementations **from scratch** $\text{no TensorFlow, no PyTorch}$ — just pure Python and NumPy. Perfect for learners who want to **deeply understand** how machine learning models work **under the hood**.

---

## \U0001F4DA Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Implementation Details](#implementation-details)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Sample Data](#sample-data)
8. [Results](#results)
9. [Extending](#extending)
10. [Project Structure](#project-structure)
11. [License](#license)
12. [Author](#author)

---

## \U0001F31F Overview

This repository builds a **fully configurable feedforward neural network** from scratch. By skipping high-level ML libraries, users will:

* Understand matrix calculus & vectorization
* Implement the full training pipeline manually
* Gain insight into how gradients propagate
* See effects of architectural choices in real-time

---

## \U0001F50D Features

* Multi-layer, fully-connected architecture
* Activation functions: Sigmoid, Tanh, ReLU
* Loss functions: MSE, Binary Cross-Entropy
* Full-batch gradient descent
* Visual logging of training loss
* Clean and modular NumPy codebase

---

## \U0001F4A1 Mathematical Foundations

### Structure of a Neural Network

A feedforward neural network consists of \$L\$ layers. Each layer \$\ell\$ performs a linear transformation followed by a non-linear activation:

$$
Z^{[\ell]} = W^{[\ell]} A^{[\ell-1]} + b^{[\ell]}\\
A^{[\ell]} = g(Z^{[\ell]})
$$

Where:

* \$W^{\[\ell]} \in \mathbb{R}^{n\_\ell \times n\_{\ell-1}}\$
* \$b^{\[\ell]} \in \mathbb{R}^{n\_\ell}\$
* \$g\$ is the activation function

### Forward Propagation

The output layer computes \$\hat{Y} = A^{\[L]}\$, where \$L\$ is the total number of layers.

### Loss Functions

**Mean Squared Error (MSE)**:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2
$$

**Binary Cross-Entropy**:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

### Backpropagation

Uses the chain rule to compute gradients layer by layer:

$$
\frac{\partial \mathcal{L}}{\partial W^{[\ell]}} = \delta^{[\ell]} {A^{[\ell-1]}}^T\\
\frac{\partial \mathcal{L}}{\partial b^{[\ell]}} = \sum_{i=1}^{m} \delta^{[\ell](i)}
$$

The error term \$\delta\$ is computed recursively:

$$
\delta^{[\ell]} = (W^{[\ell+1]})^T \delta^{[\ell+1]} \circ g'(Z^{[\ell]})
$$

### Gradient Descent

Weights and biases update as:

$$
W^{[\ell]} \leftarrow W^{[\ell]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[\ell]}}\\
b^{[\ell]} \leftarrow b^{[\ell]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[\ell]}}
$$

---

## \u2699 Implementation Details

### Initialization

* Random: \$\mathcal{N}(0,1)\$
* Xavier: \$\sqrt{\frac{1}{n\_{\text{in}}}}\$
* He: \$\sqrt{\frac{2}{n\_{\text{in}}}}\$

### Training Loop

```python
for epoch in range(epochs):
    Z[l] = W[l] @ A[l-1] + b[l]
    A[l] = activation(Z[l])
    ...
    dZ = A[L] - Y
    dW = (1/m) * dZ @ A[L-1].T
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    W[l] -= lr * dW
    b[l] -= lr * db
```

---

## \U0001F680 Installation

```bash
git clone https://github.com/humair-m/dsml.git
cd dsml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt:**

```text
numpy
matplotlib
```

---

## \U0001F3AF Usage

```bash
python "Neural Network from Scratch.py"
```

To configure:

```python
layer_sizes = [2, 4, 1]
activations = ['tanh', 'sigmoid']
learning_rate = 0.01
epochs = 1000
```

With CLI args (requires `argparse`):

```bash
python script.py --layers 2 4 1 --activations relu sigmoid --lr 0.01 --epochs 500
```

---

## \U0001F5C2 Sample Data

By default uses `sklearn.datasets.make_circles` for binary classification.

You can also use `img_.pkl`:

```python
import pickle
X, Y = pickle.load(open('img_.pkl', 'rb'))
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
```

---

## \U0001F4C8 Results

```text
Epoch 1/1000 — Loss: 0.6932
Epoch 500/1000 — Loss: 0.0513
Epoch 1000/1000 — Loss: 0.0189
```

A loss curve is saved as `loss_curve.png`.

---

## \U0001F527 Extending

* Implement Adam, RMSProp, Momentum
* Mini-batch or stochastic training
* Add Dropout or BatchNorm
* Multiclass support: Softmax + CrossEntropy

---

## \U0001F4C1 Project Structure

```
dsml/
├── Neural Network from Scratch.py
├── img_.pkl
├── requirements.txt
├── loss_curve.png
├── README.md
└── LICENSE
```

---

## \U0001F4DC License

MIT License — see [LICENSE](LICENSE) for full text.

---

## ✍️ Author

**Humair M**
GitHub: [@humair-m](https://github.com/humair-m)
Project started: June 2025
