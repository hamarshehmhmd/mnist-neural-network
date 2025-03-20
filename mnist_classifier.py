import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        Initialize the neural network
        
        Parameters:
        - input_size: Number of input features
        - hidden_sizes: List of integers for the number of neurons in each hidden layer
        - output_size: Number of output classes
        - learning_rate: Learning rate for weight updates
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize layers
        self.layers = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layers)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Initialize weights with Xavier/Glorot initialization
        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (self.layers[i] + self.layers[i+1]))
            W = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i+1]))
            b = np.zeros((1, self.layers[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Store activations and derivatives for backpropagation
        self.activations = []
        self.derivatives = []
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax activation function for output layer"""
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass through the network"""
        # Store input as the first activation
        A = X
        self.activations = [A]
        
        # Forward pass through hidden layers with sigmoid activation
        for i in range(self.num_layers - 2):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.sigmoid(Z)
            self.activations.append(A)
        
        # Output layer with softmax activation
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        output = self.softmax(Z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        """Backward pass (backpropagation)"""
        # Convert y to one-hot encoding if it's not already
        if y.ndim == 1:
            y_onehot = np.zeros((y.size, self.output_size))
            y_onehot[np.arange(y.size), y.astype(int)] = 1
        else:
            y_onehot = y
        
        # Initialize gradients
        dweights = [np.zeros_like(w) for w in self.weights]
        dbiases = [np.zeros_like(b) for b in self.biases]
        
        # Initial error (output - target)
        delta = self.activations[-1] - y_onehot
        
        # Calculate gradients for output layer
        dweights[-1] = np.dot(self.activations[-2].T, delta)
        dbiases[-1] = np.sum(delta, axis=0, keepdims=True)
        
        # Backpropagate error through hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self.sigmoid_derivative(self.activations[i+1])
            dweights[i] = np.dot(self.activations[i].T, delta)
            dbiases[i] = np.sum(delta, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dweights[i]
            self.biases[i] -= self.learning_rate * dbiases[i]
    
    def train(self, X, y, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        """Train the neural network"""
        # Store history of validation accuracy
        history = {'val_accuracy': []}
        
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, output)
            
            # Evaluate on validation set
            val_output = self.predict(X_val)
            val_accuracy = np.mean(np.argmax(val_output, axis=1) == y_val)
            history['val_accuracy'].append(val_accuracy)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}, Time: {time.time() - start_time:.2f}s")
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def predict_classes(self, X):
        """Predict class labels"""
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate the model"""
        predictions = self.predict_classes(X)
        accuracy = np.mean(predictions == y)
        return accuracy

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """Load and preprocess the MNIST dataset"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Extract labels and features
    y = df['label'].values
    X = df.drop('label', axis=1).values / 255.0  # Normalize pixel values
    
    # Split data into training (70%), validation (10%), and testing (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)  # 0.125 of 80% is 10% of the total
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to plot validation accuracy during training
def plot_training_history(history):
    """Plot validation accuracy during training"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_accuracy.png')
    plt.close()

# Function to calculate per-class accuracy
def calculate_per_class_accuracy(y_true, y_pred):
    """Calculate accuracy for each class"""
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    return per_class_accuracy

# Main function
def main():
    # Load and preprocess data
    file_path = "mnist_data.csv"
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(file_path)
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Network parameters
    input_size = X_train.shape[1]  # Number of features (pixels)
    hidden_sizes = [128]  # Number of neurons in hidden layer(s) - can be changed easily
    output_size = 10  # 10 digits (0-9)
    learning_rate = 0.01
    epochs = 50
    batch_size = 64
    
    # Create and train the neural network
    nn = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate)
    print(f"Neural Network Structure:")
    print(f"Input Layer: {input_size} neurons")
    for i, size in enumerate(hidden_sizes):
        print(f"Hidden Layer {i+1}: {size} neurons")
    print(f"Output Layer: {output_size} neurons")
    
    print("\nTraining the neural network...")
    history = nn.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    # Plot validation accuracy during training
    plot_training_history(history)
    
    # Evaluate the model on test set
    test_accuracy = nn.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Calculate per-class accuracy
    y_pred = nn.predict_classes(X_test)
    per_class_accuracy = calculate_per_class_accuracy(y_test, y_pred)
    
    print("\nPer-class accuracy:")
    for i, accuracy in enumerate(per_class_accuracy):
        print(f"Class {i}: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main() 