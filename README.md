# Handwritten Digit Recognition — Neural Network from Scratch

A neural network that classifies handwritten digits (0–9) from the MNIST dataset, built from first principles in Python using only **NumPy** and **pandas** — no PyTorch, TensorFlow or scikit-learn.

**Result: 97% accuracy on unseen test data.**

<!-- Optional: add an image here, e.g. a training-loss curve or sample predictions -->
<!-- ![Training loss](images/loss_curve.png) -->

## Why I built it

I wanted to understand what actually happens inside a neural network rather than calling a library. Writing every step by hand — the forward pass, the gradients, and the parameter updates — meant deriving and implementing the maths of backpropagation myself.

## How it works

**Data**
- MNIST: 28×28 greyscale images of handwritten digits, flattened into 784-length input vectors
- Loaded with pandas, with pixel values normalised to [0, 1] and labels one-hot encoded

**Architecture**
- Input layer: 784 neurons
- Hidden layer(s): `[FILL IN, e.g. 128 neurons, ReLU activation]`
- Output layer: 10 neurons with softmax, giving a probability for each digit

**Training**
- **Forward propagation:** each layer computes `Z = W·A + b`, followed by its activation function
- **Loss:** cross-entropy between the predicted probabilities and the true label
- **Backpropagation:** gradients of the loss with respect to every weight and bias, derived with the chain rule and implemented in vectorised NumPy
- **Mini-batch stochastic gradient descent:** the training data is shuffled and split into small batches, with parameters updated after each batch: `W ← W − α · ∂L/∂W`
- Hyperparameters: learning rate `[FILL IN]`, batch size `[FILL IN]`, epochs `[FILL IN]`

## Results

| Metric | Value |
|---|---|
| Test accuracy (unseen data) | **97%** |
| Training time | `[FILL IN, optional]` |

## How to run

```bash
git clone https://github.com/[YOUR-USERNAME]/[REPO-NAME].git
cd [REPO-NAME]
pip install numpy pandas
python [MAIN-FILE].py
```

The MNIST data is `[FILL IN: where to get it, e.g. "downloaded from Kaggle as train.csv / test.csv and placed in a data/ folder"]`.

## What I learned

- How backpropagation follows from the chain rule, and how to vectorise it efficiently with matrices
- Why mini-batches give a good balance between noisy single-sample updates and slow full-batch updates
- Practical issues in training: weight initialisation, choosing a learning rate, and checking for overfitting with held-out data

## Possible extensions

- Add momentum or the Adam optimiser
- Add regularisation (L2 or dropout)
- Try a convolutional architecture and compare accuracy
