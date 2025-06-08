# dsml

A hands-on repository of **Data Science** and **Machine Learning** models built entirely **from scratch** — no TensorFlow, no PyTorch — just pure Python 🐍 and NumPy 📊. Ideal for learners who want to dig into the **core mechanics** of how machine learning actually works.

---

## 📚 Table of Contents

1. [🌟 Overview](#-overview)
2. [🔍 Features](#-features)
3. [🧠 Mathematical Foundations](#-mathematical-foundations)
4. [⚙️ Implementation Details](#-implementation-details)
5. [🚀 Installation](#-installation)
6. [🎯 Usage](#-usage)
7. [🗂 Sample Data](#-sample-data)
8. [📈 Results](#-results)
9. [🛠 Extending](#-extending)
10. [📁 Project Structure](#-project-structure)
11. [📜 License](#-license)
12. [✍️ Author](#️-author)

---

## 🌟 Overview

This repo builds a **configurable feedforward neural network** from the ground up. By avoiding high-level libraries, you’ll:

* Understand matrix calculus & vectorized ops 📐
* Manually code each training step 🧪
* Observe how gradients propagate 🔁
* Experiment with layers and activations 🔧

---

## 🔍 Features

* ✅ Fully-connected multi-layer architecture
* ⚡ Activation functions: Sigmoid, Tanh, ReLU
* 💥 Loss functions: MSE, Binary Cross-Entropy
* 🌀 Full-batch gradient descent optimizer
* 📊 Visual loss tracking via `matplotlib`
* 💡 Clean, modular NumPy code

---

## 🧠 Mathematical Foundations

### 🧱 Neural Network Layers

Each layer $\ell$ performs:

$$
Z^{[\ell]} = W^{[\ell]} A^{[\ell-1]} + b^{[\ell]} \\
A^{[\ell]} = g(Z^{[\ell]})
$$

Where:

* $W^{[\ell]}$ = weight matrix
* $b^{[\ell]}$ = bias vector
* $g$ = activation function

### ➕ Forward Propagation

Final prediction:

$$
\hat{Y} = A^{[L]}
$$

### ❌ Loss Functions

**Mean Squared Error (MSE):**

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2
$$

**Binary Cross-Entropy (BCE):**

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

### 🔁 Backpropagation

Uses chain rule to compute gradients:

$$
\frac{\partial \mathcal{L}}{\partial W^{[\ell]}} = \delta^{[\ell]} {A^{[\ell-1]}}^T \\
\frac{\partial \mathcal{L}}{\partial b^{[\ell]}} = \sum_{i=1}^{m} \delta^{[\ell](i)}
$$

Error is computed recursively:

$$
\delta^{[\ell]} = (W^{[\ell+1]})^T \delta^{[\ell+1]} \circ g'(Z^{[\ell]})
$$

### 📉 Gradient Descent

Parameter update rule:

$$
W^{[\ell]} \leftarrow W^{[\ell]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[\ell]}} \\
b^{[\ell]} \leftarrow b^{[\ell]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[\ell]}}
$$

---

## ⚙️ Implementation Details

### 🧪 Initialization Methods

* **Random**: $\mathcal{N}(0,1)$
* **Xavier**: $\sqrt{\frac{1}{n_{in}}}$
* **He**: $\sqrt{\frac{2}{n_{in}}}$

### 🔄 Training Loop

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

## 🚀 Installation

```bash
git clone https://github.com/humair-m/dsml.git
cd dsml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**`requirements.txt`**

```
numpy
matplotlib
```

---

## 🎯 Usage

To run the main script:

```bash
python "Neural Network from Scratch.py"
```

To modify configuration:

```python
layer_sizes = [2, 4, 1]
activations = ['tanh', 'sigmoid']
learning_rate = 0.01
epochs = 1000
```

Run with CLI arguments (if using `argparse`):

```bash
python script.py --layers 2 4 1 --activations relu sigmoid --lr 0.01 --epochs 500
```

---

## 🗂 Sample Data

Uses `make_circles` from `sklearn.datasets` by default.

Optionally, load your own dataset like `img_.pkl`:

```python
import pickle
X, Y = pickle.load(open('img_.pkl', 'rb'))
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
```

---

## 📈 Results

```text
Epoch 1/1000 — Loss: 0.6932
Epoch 500/1000 — Loss: 0.0513
Epoch 1000/1000 — Loss: 0.0189
```

🖼 A loss curve is saved as `loss_curve.png`

---

## 🛠 Extending

* Add optimizers: Adam, RMSProp, Momentum
* Implement mini-batch or stochastic training
* Use regularization: Dropout, BatchNorm
* Add multiclass support via Softmax + CrossEntropy

---

## 📁 Project Structure

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

## 📜 License

MIT License — see [LICENSE](LICENSE) for full terms.

---

## ✍️ Author

**Humair M**
GitHub: [@humair-m](https://github.com/humair-m)
📅 Project started: June 2025
