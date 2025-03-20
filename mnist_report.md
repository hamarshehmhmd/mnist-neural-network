# MNIST Classification using Neural Networks with Backpropagation

## Neural Network Structure

The Artificial Neural Network (ANN) implemented for this assignment has the following structure:

1. **Input Layer**: 784 neurons (28×28 pixels of MNIST images)
2. **Hidden Layer**: 128 neurons with sigmoid activation function
3. **Output Layer**: 10 neurons (one for each digit 0-9) with softmax activation

This structure can be easily modified by changing the `hidden_sizes` parameter in the code:
```python
# Network parameters
hidden_sizes = [128]  # Number of neurons in hidden layer(s)
```

You can increase the number of hidden layers by adding more values to this list, for example:
```python
hidden_sizes = [128, 64]  # Two hidden layers with 128 and 64 neurons respectively
```

## Weight Update Equations

The neural network uses backpropagation with gradient descent to update the weights. The following equations describe the weight update process:

### Forward Propagation

For each layer l (from input to output):

1. Calculate weighted sum:
   Z^(l) = A^(l-1) · W^(l) + b^(l)

2. Apply activation function:
   - For hidden layers: A^(l) = sigmoid(Z^(l)) = 1 / (1 + e^(-Z^(l)))
   - For output layer: A^(output) = softmax(Z^(output)) = e^(Z_i) / Σe^(Z_j)

Where:
- Z^(l) is the weighted sum for layer l
- A^(l) is the activation output for layer l
- W^(l) are the weights for layer l
- b^(l) are the biases for layer l

### Backward Propagation

1. Calculate output layer error:
   δ^(output) = A^(output) - y

   Where y is the one-hot encoded target vector.

2. Calculate hidden layer error:
   δ^(l) = (δ^(l+1) · W^(l+1)^T) ⊙ sigmoid_derivative(A^(l))

   Where ⊙ represents element-wise multiplication and sigmoid_derivative(A) = A * (1-A)

3. Calculate gradients:
   - dW^(l) = A^(l-1)^T · δ^(l)
   - db^(l) = sum(δ^(l), axis=0)

4. Update weights and biases:
   - W^(l) = W^(l) - η * dW^(l)
   - b^(l) = b^(l) - η * db^(l)

   Where η (eta) is the learning rate.

## Expected Results

### Test Accuracy

The neural network is expected to achieve at least 80% accuracy on the test set (20% of the total dataset). The actual test accuracy will be reported after running the implementation.

### Per-Class Accuracy

The accuracy for each digit class (0-9) will be calculated and reported separately to show how well the model performs for different digits.

### Validation Accuracy During Training

A plot showing the validation accuracy progression throughout the training process will be generated and saved as `validation_accuracy.png`. This visualization helps in understanding the learning behavior of the network and identifying potential issues like overfitting.

## Implementation Details

- The dataset is split into training (70%), validation (10%), and testing (20%) sets.
- Mini-batch gradient descent is used for better convergence.
- Xavier/Glorot initialization is used for weight initialization to help with faster convergence.
- Learning rate is set to 0.01, which can be adjusted as needed.
- The implementation uses 50 epochs of training with a batch size of 64.

## How to Run the Code

To run the implementation:

```
python mnist_classifier.py
```

This will train the neural network on the MNIST dataset, display the accuracy on the test set, generate the validation accuracy plot, and show per-class accuracies. 