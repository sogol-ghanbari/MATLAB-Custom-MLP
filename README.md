# Multilayer Perceptron (MLP) from Scratch (MATLAB)

A custom, ground-up implementation of a **Multilayer Perceptron (MLP)** neural network in MATLAB for non-linear dynamic system identification and function approximation.

## Overview
This repository contains a pure MATLAB implementation of an MLP architecture trained using backpropagation and gradient descent. Built entirely without dependencies on high-level machine learning libraries or the MATLAB Deep Learning Toolbox, it serves as a transparent reference for understanding the inner mechanics of feedforward neural networks.

## Key Features
- **From-Scratch Implementation:** Full forward pass, loss computation, and backpropagation derived and coded manually.
- **Dynamic System Modeling:** Evaluated on non-linear discrete-time dynamic systems.
- **Validation Pipeline:** Structured train/test splits to monitor convergence and prevent overfitting.
- **Analytical Derivatives:** Exact mathematical gradient formulation across hidden and output layers.
- **Live Convergence Visuals:** Tracks Mean Squared Error (MSE) progression across training epochs.

## Architecture & Settings
- **Network Topology:** Multi-input feedforward architecture with configurable hidden nodes.
- **Activation Functions:** Hyperbolic tangent (`tanh` / `tansig`) for non-linear hidden representations, linear (`purelin`) for output projection.
- **Optimization:** Mini-batch/Sample-based Gradient Descent with adaptive monitoring.

## How to Run
1. Clone or download this repository.
2. Open the `.m` script in MATLAB.
3. Execute the code to run system data generation, train the network, and inspect final approximation figures.

## License
This project is open-source and available under the [MIT License](LICENSE).

