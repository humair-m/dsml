import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DigitClassifier:
    """
    A neural network classifier for handwritten digit recognition.
    
    Features:
    - 3-layer neural network with ReLU activation
    - Dropout regularization
    - Gradient clipping
    - Batch training
    - Model persistence
    """
    
    def __init__(self, input_size: int = 784, hidden_size1: int = 128, 
                 hidden_size2: int = 64, output_size: int = 10, 
                 learning_rate: float = 1e-3, dropout_rate: float = 0.2):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of input layer (default: 784 for 28x28 images)
            hidden_size1: Size of first hidden layer
            hidden_size2: Size of second hidden layer
            output_size: Size of output layer (number of classes)
            learning_rate: Learning rate for training
            dropout_rate: Dropout rate for regularization
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        self._initialize_weights()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.trained = False
        
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size1))
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * np.sqrt(2.0 / self.hidden_size1)
        self.b2 = np.zeros((1, self.hidden_size2))
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * np.sqrt(2.0 / self.hidden_size2)
        self.b3 = np.zeros((1, self.output_size))
    
    @staticmethod
    def relu(x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU activation function."""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x):
        """Softmax activation function with numerical stability."""
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def forward(self, X, training=False):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data (shape: [batch_size, input_size])
            training: Whether in training mode (affects dropout)
            
        Returns:
            Tuple of (z1, a1, z2, a2, z3, a3)
        """
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected input shape [batch_size, {self.input_size}], got {X.shape}")
        
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        
        if training:
            dropout_mask1 = np.random.rand(*a1.shape) > self.dropout_rate
            a1 *= dropout_mask1
            a1 /= (1 - self.dropout_rate)
        
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        
        if training:
            dropout_mask2 = np.random.rand(*a2.shape) > self.dropout_rate
            a2 *= dropout_mask2
            a2 /= (1 - self.dropout_rate)
        
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.softmax(z3)
        
        return z1, a1, z2, a2, z3, a3
    
    def backward(self, X, y, z1, a1, z2, a2, z3, a3, gradient_clip_value=1.0):
        """
        Backward propagation to compute gradients and update weights.
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            z1, a1, z2, a2, z3, a3: Activations from forward pass
            gradient_clip_value: Value for gradient clipping
        """
        m = X.shape[0]
        
        dz3 = a3 - y
        dz3 = np.clip(dz3, -gradient_clip_value, gradient_clip_value)
        
        dW3 = (1/m) * np.dot(a2.T, dz3)
        db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
        
        dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(z2)
        dz2 = np.clip(dz2, -gradient_clip_value, gradient_clip_value)
        
        dW2 = (1/m) * np.dot(a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(z1)
        dz1 = np.clip(dz1, -gradient_clip_value, gradient_clip_value)
        
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss."""
        return -np.mean(y_true * np.log(y_pred + 1e-10))
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, gradient_clip_value=1.0, verbose=True):
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            gradient_clip_value: Value for gradient clipping
            verbose: Whether to print training progress
        """
        if X_train.shape[1] != self.input_size:
            raise ValueError(f"X_train shape {X_train.shape} incompatible with input_size {self.input_size}")
        if y_train.shape[1] != self.output_size:
            raise ValueError(f"y_train shape {y_train.shape} incompatible with output_size {self.output_size}")
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                z1, a1, z2, a2, z3, a3 = self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch, z1, a1, z2, a2, z3, a3, gradient_clip_value)
            
            _, _, _, _, _, train_pred = self.forward(X_train, training=False)
            train_loss = self.compute_loss(y_train, train_pred)
            self.train_losses.append(train_loss)
            
            if X_val is not None and y_val is not None:
                _, _, _, _, _, val_pred = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val, val_pred)
                self.val_losses.append(val_loss)
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}')
        
        self.trained = True
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions on input data."""
        if not self.trained:
            print("Warning: Model hasn't been trained yet!")
        
        _, _, _, _, _, output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.trained:
            print("Warning: Model hasn't been trained yet!")
        
        _, _, _, _, _, output = self.forward(X, training=False)
        return output
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        predictions = self.predict(X_test)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(true_labels, predictions)
        
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        return accuracy
    
    def save_model(self, filepath="digit_classifier_model.pkl"):
        """Save the trained model."""
        model_data = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'input_size': self.input_size,
            'hidden_size1': self.hidden_size1,
            'hidden_size2': self.hidden_size2,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="digit_classifier_model.pkl"):
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']
        self.W3 = model_data['W3']
        self.b3 = model_data['b3']
        
        self.input_size = model_data['input_size']
        self.hidden_size1 = model_data['hidden_size1']
        self.hidden_size2 = model_data['hidden_size2']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.dropout_rate = model_data['dropout_rate']
        
        self.train_losses = model_data.get('train_losses', [])
        self.val_losses = model_data.get('val_losses', [])
        self.trained = model_data.get('trained', True)
        
        print(f"Model loaded from {filepath}")


class DataLoader:
    """Utility class for loading and preprocessing digit data."""
    
    @staticmethod
    def load_from_pickle(pickle_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data from pickle file."""
        print(f"Loading data from pickle file: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, tuple) and len(data) == 4:
            X_train, X_test, y_train, y_test = data
        else:
            raise ValueError("Pickle file should contain (X_train, X_test, y_train, y_test)")
            
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Ensure data is 2D
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        if X_train.max() > 1.0:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
            
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_from_directories(data_dir: str, image_size: Tuple[int, int] = (28, 28), test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data from directory structure (digit folders)."""
        print(f"Loading data from directories: {data_dir}")
        
        X, y = [], []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Directory {data_dir} does not exist!")
        
        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue
                
            try:
                class_label = int(class_dir.name)
            except ValueError:
                print(f"Skipping non-numeric directory: {class_dir.name}")
                continue
                
            print(f"Processing class: {class_label}")
            
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    continue
                
                try:
                    img = Image.open(img_file).convert('L')
                    img = np.array(img)
                    img = cv2.resize(img, image_size)
                    img = img / 255.0
                    X.append(img.flatten())
                    y.append(class_label)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        
        X, y = np.array(X), np.array(y)
        print(f"Loaded {len(X)} images")
        
        if len(X) == 0:
            raise ValueError("No images were loaded! Check your directory structure and file formats.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def preprocess_labels(y_train, y_test, num_classes=10):
        """Convert labels to one-hot encoding."""
        encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
        
        return y_train_encoded, y_test_encoded, encoder
    
    @staticmethod
    def save_processed_data(X_train, X_test, y_train, y_test, filepath="processed_data.pkl"):
        """Save processed data to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Processed data saved to {filepath}")

    @staticmethod
    def load_mnist_like_data(input_size=784):
        """Create sample MNIST-like data for testing purposes."""
        print("Creating sample MNIST-like data for testing...")
        
        np.random.seed(42)
        n_samples = 1000
        X = np.random.rand(n_samples, input_size)
        y = np.random.randint(0, 10, n_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Generated {len(X_train)} training samples and {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test


class Visualizer:
    """Utility class for visualization."""
    
    @staticmethod
    def plot_training_history(train_losses, val_losses=None, save_path=None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def show_sample_predictions(X, y_true, model, num_samples=20, image_shape=(28, 28), save_path=None):
        """Display exactly 20 sample predictions with real and predicted labels."""
        if len(X) < num_samples:
            print(f"Warning: Only {len(X)} samples available, displaying all.")
            num_samples = len(X)
        
        predictions = model.predict(X[:num_samples])
        true_labels = np.argmax(y_true[:num_samples], axis=1)
        
        # Reshape images for display
        try:
            images = X[:num_samples].reshape(-1, *image_shape)
        except ValueError as e:
            print(f"Error reshaping images: {e}. Check if input_size matches image_shape ({image_shape[0]*image_shape[1]}).")
            return
        
        # Create 4x5 subplot grid for 20 images
        rows, cols = 4, 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()  # Flatten for easier iteration
        
        for i in range(num_samples):
            axes[i].imshow(images[i], cmap='gray')
            color = 'green' if predictions[i] == true_labels[i] else 'red'
            axes[i].set_title(f'Real: {true_labels[i]}, Pred: {predictions[i]}', 
                             color=color, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        # Turn off any unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Real vs Predicted Labels)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        accuracy = np.mean(predictions == true_labels) * 100
        print(f"Sample Accuracy (first {num_samples} samples): {accuracy:.2f}%")
        for i in range(num_samples):
            print(f"Image {i+1}: Real Label = {true_labels[i]}, Predicted Label = {predictions[i]}")


def main():
    """Main function demonstrating how to use the digit classifier."""
    
    # Configuration
    USE_PICKLE = True
    PICKLE_PATH = "/home/humair/Downloads/img_.pkl"
    DATA_DIR = "/home/humair/digit_images"
    USE_SAMPLE_DATA = False
    IMAGE_SIZE = (54, 54)  # Matches your 2916 features (54x54)
    
    try:
        # Load data
        if USE_SAMPLE_DATA:
            X_train, X_test, y_train_raw, y_test_raw = DataLoader.load_mnist_like_data(input_size=IMAGE_SIZE[0] * IMAGE_SIZE[1])
        elif USE_PICKLE and os.path.exists(PICKLE_PATH):
            X_train, X_test, y_train_raw, y_test_raw = DataLoader.load_from_pickle(PICKLE_PATH)
        elif os.path.exists(DATA_DIR):
            X_train, X_test, y_train_raw, y_test_raw = DataLoader.load_from_directories(DATA_DIR, image_size=IMAGE_SIZE)
        else:
            print("No data source found. Using sample data.")
            X_train, X_test, y_train_raw, y_test_raw = DataLoader.load_mnist_like_data(input_size=IMAGE_SIZE[0] * IMAGE_SIZE[1])
        
        # Detect input size
        input_size = X_train.shape[1]
        print(f"Detected input size: {input_size}")
        
        # Preprocess labels
        y_train, y_test, encoder = DataLoader.preprocess_labels(y_train_raw, y_test_raw)
        
        print(f"Data loaded successfully!")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Input features: {input_size}")
        print(f"Number of classes: {y_train.shape[1]}")
        
        # Model configuration
        config = {
            'hidden_size1': 128,
            'hidden_size2': 64,
            'learning_rate': 1e-3,
            'dropout_rate': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'gradient_clip_value': 1.0
        }
        
        # Create train/validation split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Initialize model
        model = DigitClassifier(
            input_size=input_size,
            output_size=y_train.shape[1],
            **{k: v for k, v in config.items() if k not in ['epochs', 'batch_size', 'gradient_clip_value']}
        )
        
        print("Starting training...")
        model.train(
            X_train_split, y_train_split, X_val, y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            gradient_clip_value=config['gradient_clip_value']
        )
        
        # Save model
        model.save_model("/home/humair/digit_classifier_model.pkl")
        
        # Evaluate model
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        train_accuracy = model.evaluate(X_train, y_train)
        test_accuracy = model.evaluate(X_test, y_test)
        
        # Visualizations
        print("\nGenerating visualizations...")
        Visualizer.plot_training_history(
            model.train_losses, model.val_losses, 
            save_path="/home/humair/training_history.png"
        )
        
        # Display 20 sample predictions
        image_shape = IMAGE_SIZE if input_size == IMAGE_SIZE[0] * IMAGE_SIZE[1] else None
        if image_shape:
            print("\nDisplaying 20 sample predictions...")
            Visualizer.show_sample_predictions(
                X_test, y_test, model, num_samples=20, image_shape=image_shape,
                save_path="/home/humair/sample_predictions.png"
            )
        else:
            print("Cannot display images: input size does not match expected image shape.")
        
        # Plot confusion matrix
        test_predictions = model.predict(X_test)
        test_true = np.argmax(y_test, axis=1)
        Visualizer.plot_confusion_matrix(
            test_true, test_predictions,
            save_path="/home/humair/confusion_matrix.png"
        )
        
        print("Training and evaluation completed successfully!")
        print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that your data paths are correct")
        print("2. Ensure your pickle file contains (X_train, X_test, y_train, y_test)")
        print("3. For directory loading, ensure structure is: DATA_DIR/0/, DATA_DIR/1/, ... DATA_DIR/9/")
        print("4. Set USE_SAMPLE_DATA = True to test with generated data")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
