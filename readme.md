# Multiple linear regression

This project implements a multivariate linear regression model from scratch in pure Rust.

## How it works

1. Load data 
    - Read the CSV
2. Preprocess data
    - Split data 80/20 and Z-score normalize
3. Train model   
    - Initialize random weights and a bias with value 0
    - Minimize mean squared error (MSE) with gradient descent
    - Iterate over epochs and update weights and bias
4. Predict and evaluate
    - Make predictions on test data
    - Calculate mean squared error to evaluate model performance

## Mathematics

Linear regression: `ŷ = w₁x₁ + w₂x₂ + w₃x₃ + b`

Loss function: `MSE = (1/n) * Σ(y - ŷ)²`

Gradient descent updates: `wᵢ = wᵢ - α * (∂MSE/∂wᵢ)` `b = b - α * (∂MSE/∂b)`

`ŷ` = predicted value
`w` = weights
`b` = bias term
`α` = learning rate
`n` = number of samples
