import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

# Load and Preprocess the Dataset
def load_data(data_dir=None, use_pickle=True, pickle_path="/kaggle/input/digits/img_.pkl"):
    """
    Load data either from a pickle file or from image directories
    
    Args:
        data_dir: Directory containing subfolders (each named as a digit class)
        use_pickle: Boolean to determine if pickle file should be used
        pickle_path: Path to pickle file if use_pickle is True
    
    Returns:
        X_train, X_test, y_train, y_test, num_classes
    """
    if use_pickle:
        print("Loading data from pickle file...")
        with open(pickle_path, 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    else:
        print(f"Processing images from directory: {data_dir}")
        X, y = [], []
        
        # Iterate through each subdirectory (each class)
        for class_dir in sorted(os.listdir(data_dir)):
            class_path = os.path.join(data_dir, class_dir)
            
            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue
                
            class_label = int(class_dir)  # Assuming directory names are digits
            print(f"Processing class: {class_label}")
            
            # Process all images in the class directory
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                
                # Skip non-image files
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                
                # Load and preprocess image
                try:
                    # Using PIL for initial loading
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = np.array(img)
                    
                    # Resize to a fixed size (28x28) - typical for digit recognition
                    img = cv2.resize(img, (28, 28))
                    
                    # Normalize pixel values to 0-1
                    img = img / 255.0
                    
                    X.append(img.flatten())  # Flatten the image
                    y.append(class_label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        X, y = np.array(X), np.array(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape data
    X_train = np.array(X_train).reshape(len(X_train), -1)
    X_test = np.array(X_test).reshape(len(X_test), -1)
    
    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test = encoder.transform(np.array(y_test).reshape(-1, 1))
    
    num_classes = len(encoder.categories_[0])
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Number of classes: {num_classes}")
    
    return X_train, X_test, y_train, y_test, num_classes

# Function to save processed data
def save_processed_data(X_train, X_test, y_train, y_test, output_path="processed_data.pkl"):
    """Save the processed data to a pickle file for future use"""
    with open(output_path, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    print(f"Data saved to {output_path}")

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Initialize Weights
def initialize_weights(input_size, hidden_size1, hidden_size2, output_size):
    np.random.seed(42)
    weights_input_hidden1 = np.random.randn(input_size, hidden_size1) * 0.01
    bias_hidden1 = np.zeros((1, hidden_size1))
    weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
    bias_hidden2 = np.zeros((1, hidden_size2))
    weights_hidden2_output = np.random.randn(hidden_size2, output_size) * 0.01
    bias_output = np.zeros((1, output_size))
    return (weights_input_hidden1, bias_hidden1, 
            weights_hidden1_hidden2, bias_hidden2, 
            weights_hidden2_output, bias_output)

# Forward Propagation
def forward(X, weights, dropout=False, dropout_rate=0.2):
    weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output = weights

    hidden_layer1_input = np.dot(X, weights_input_hidden1) + bias_hidden1
    hidden_layer1 = relu(hidden_layer1_input)
    if dropout:
        dropout_mask1 = np.random.rand(*hidden_layer1.shape) > dropout_rate
        hidden_layer1 *= dropout_mask1
        hidden_layer1 /= (1 - dropout_rate)

    hidden_layer2_input = np.dot(hidden_layer1, weights_hidden1_hidden2) + bias_hidden2
    hidden_layer2 = relu(hidden_layer2_input)
    if dropout:
        dropout_mask2 = np.random.rand(*hidden_layer2.shape) > dropout_rate
        hidden_layer2 *= dropout_mask2
        hidden_layer2 /= (1 - dropout_rate)

    output_layer_input = np.dot(hidden_layer2, weights_hidden2_output) + bias_output
    output_layer = softmax(output_layer_input)
    return hidden_layer1, hidden_layer2, output_layer

# Backward Propagation
def backward(X, y, hidden_layer1, hidden_layer2, output, weights, learning_rate, gradient_clip_value):
    weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output = weights

    output_error = output - y
    output_error = np.clip(output_error, -gradient_clip_value, gradient_clip_value)

    hidden2_error = np.dot(output_error, weights_hidden2_output.T) * relu_derivative(hidden_layer2)
    hidden2_error = np.clip(hidden2_error, -gradient_clip_value, gradient_clip_value)

    hidden1_error = np.dot(hidden2_error, weights_hidden1_hidden2.T) * relu_derivative(hidden_layer1)
    hidden1_error = np.clip(hidden1_error, -gradient_clip_value, gradient_clip_value)

    weights_hidden2_output -= learning_rate * np.dot(hidden_layer2.T, output_error)
    bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

    weights_hidden1_hidden2 -= learning_rate * np.dot(hidden_layer1.T, hidden2_error)
    bias_hidden2 -= learning_rate * np.sum(hidden2_error, axis=0, keepdims=True)

    weights_input_hidden1 -= learning_rate * np.dot(X.T, hidden1_error)
    bias_hidden1 -= learning_rate * np.sum(hidden1_error, axis=0, keepdims=True)

    return (weights_input_hidden1, bias_hidden1, 
            weights_hidden1_hidden2, bias_hidden2, 
            weights_hidden2_output, bias_output)

# Compute Loss
def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-10))

# Training the Model
def train(X_train, y_train, X_val, y_val, weights, epochs, batch_size, learning_rate, gradient_clip_value):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            hidden_layer1, hidden_layer2, output = forward(X_batch, weights, dropout=True)
            weights = backward(X_batch, y_batch, hidden_layer1, hidden_layer2, output, weights, learning_rate, gradient_clip_value)

        # Compute and store losses
        _, _, output_train = forward(X_train, weights)
        _, _, output_val = forward(X_val, weights)
        train_loss = compute_loss(y_train, output_train)
        val_loss = compute_loss(y_val, output_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    return weights, train_losses, val_losses

# Save Model Weights
def save_model(weights, model_path="model_weights.pkl"):
    """Save the model weights to a file"""
    with open(model_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Model saved to {model_path}")

# Load Model Weights
def load_model(model_path="model_weights.pkl"):
    """Load model weights from a file"""
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return weights

# Plot Losses
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curve.png")
    plt.show()

# Show Predictions
def show_predictions(X, y, weights, index=None, save_fig=False):
    _, _, output = forward(X, weights)
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    image_size = int(np.sqrt(X.shape[1]))
    reshaped_images = X.reshape(-1, image_size, image_size)

    if index is not None:
        plt.figure(figsize=(4, 4))
        plt.imshow(reshaped_images[index], cmap='gray')
        plt.title(f'Pred: {predictions[index]}, True: {labels[index]}')
        plt.axis('off')
        if save_fig:
            plt.savefig(f"prediction_sample_{index}.png")
        plt.show()
    else:
        plt.figure(figsize=(12, 8))
        for i in range(20):
            plt.subplot(4, 5, i + 1)
            plt.imshow(reshaped_images[i], cmap='gray')
            plt.title(f'P: {predictions[i]}, T: {labels[i]}')
            plt.axis('off')
        plt.tight_layout()
        if save_fig:
            plt.savefig("prediction_samples.png")
        plt.show()

# Predict new image
def predict_image(image_path, weights):
    """Predict the digit in a single image"""
    try:
        # Load and preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = np.array(img)
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        img = img / 255.0  # Normalize
        
        # Reshape for prediction
        img = img.reshape(1, -1)
        
        # Predict
        _, _, output = forward(img, weights)
        prediction = np.argmax(output, axis=1)[0]
        confidence = np.max(output) * 100
        
        # Display the image and prediction
        plt.figure(figsize=(5, 5))
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'Prediction: {prediction} (Confidence: {confidence:.2f}%)')
        plt.axis('off')
        plt.show()
        
        return prediction, confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# Process a directory of new images
def process_directory(directory_path, weights):
    """Process all images in a directory and show predictions"""
    results = {}
    
    for img_file in os.listdir(directory_path):
        img_path = os.path.join(directory_path, img_file)
        
        # Skip non-image files
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
            
        print(f"Processing {img_file}...")
        prediction, confidence = predict_image(img_path, weights)
        
        if prediction is not None:
            results[img_file] = (prediction, confidence)
    
    return results

# Main Execution
if __name__ == '__main__':
    # Configuration
    DATA_DIR = "digit_images"  # Directory containing subdirectories of digit classes
    USE_PICKLE = False  # Set to False to process images from directories
    PICKLE_PATH = "/kaggle/input/digits/img_.pkl"  # Path to pickle file if USE_PICKLE is True
    
    # Model parameters
    hidden_size1, hidden_size2 = 156, 128
    learning_rate = 1e-5
    gradient_clip_value = 1.0
    batch_size = 32
    epochs = 50
    
    # Load data
    X_train, X_test, y_train, y_test, output_size = load_data(
        data_dir=DATA_DIR, 
        use_pickle=USE_PICKLE, 
        pickle_path=PICKLE_PATH
    )
    
    # You can save the processed data for future use
    save_processed_data(X_train, X_test, y_train, y_test, "processed_digits.pkl")
    
    # Initialize the model
    input_size = X_train.shape[1]
    weights = initialize_weights(input_size, hidden_size1, hidden_size2, output_size)
    
    # Split training data into train and validation
    split_index = int(0.8 * len(X_train))
    X_train_split, X_val = X_train[:split_index], X_train[split_index:]
    y_train_split, y_val = y_train[:split_index], y_train[split_index:]

    # Train the model
    weights, train_losses, val_losses = train(
        X_train_split, y_train_split, X_val, y_val, 
        weights, epochs, batch_size, learning_rate, gradient_clip_value
    )
    
    # Save the trained model
    save_model(weights, "digit_classifier_model.pkl")
    
    # Evaluate the model
    print("Evaluating model...")
    _, _, output_train = forward(X_train, weights)
    _, _, output_test = forward(X_test, weights)
    train_accuracy = np.mean(np.argmax(output_train, axis=1) == np.argmax(y_train, axis=1)) * 100
    test_accuracy = np.mean(np.argmax(output_test, axis=1) == np.argmax(y_test, axis=1)) * 100
    
    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Plot the training and validation losses
    plot_loss(train_losses, val_losses)
    
    # Show some example predictions
    print("Showing sample predictions...")
    show_predictions(X_test, y_test, weights, save_fig=True)
    
    # To process a new directory of images:
    # new_images_dir = "new_digit_images"
    # if os.path.exists(new_images_dir):
    #     print(f"\nProcessing new images from {new_images_dir}...")
    #     results = process_directory(new_images_dir, weights)
    #     for img_file, (pred, conf) in results.items():
    #         print(f"{img_file}: Predicted {pred} with {conf:.2f}% confidence")
