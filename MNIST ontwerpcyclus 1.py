import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import time
import pickle
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# load the mnist dataset
mnist_data, mnist_labels = fetch_openml(
    "mnist_784", version=1, return_X_y=True, parser="auto")

mnist_data = np.asarray(mnist_data, dtype=np.float32)
mnist_labels = np.asarray(mnist_labels, dtype=np.int32)

data_train, data_test, labels_train, labels_test = train_test_split(
    mnist_data, mnist_labels, test_size=10000, random_state=42)

# Dividing the data into a subset
data_train = data_train[:40000] / 255.0
data_train = data_train.reshape(-1, 28, 28)

data_test = data_test[:10000] / 255.0
data_test = data_test.reshape(-1, 28, 28)

labels_train = labels_train[:40000]

labels_test = labels_test[:10000] 

resume_training = True
start_epoch = 0
checkpoint_dir = 'checkpoints_ontwerpcyclus_1/'
selected_checkpoint = (f"{checkpoint_dir}model_parameters_epoch_24.pkl")

# Check if the directory exists; if not, create it
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

class Convolution:

    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape

        # Size of outputs and kernels

        self.filter_shape = (num_filters, filter_size, filter_size)  # (6,6)
        self.output_shape = (num_filters, input_height -
                             filter_size + 1, input_width - filter_size + 1)

        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_data):
        self.input_data = input_data

        output = np.zeros(self.output_shape) # Array filled with zeros with the same shape as our output
        for i in range(self.num_filters):
            output[i] = correlate2d(
                self.input_data, self.filters[i], mode="valid")
        # ReLU activation function
        output = np.maximum(0, output)
        return output

    def backward(self, der_out, learning_rate):
        der_input = np.zeros_like(self.input_data)
        der_filters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
            # Calculating gradient loss
            der_filters[i] = correlate2d(
                self.input_data, der_out[i], mode="valid")
            der_input += correlate2d(der_out[i], self.filters[i], mode="full")

        # Updating filters and biases with learning rate
        self.filters -= learning_rate * der_filters
        self.biases -= learning_rate * der_out

        return der_input
    
    def get_parameters(self):
        return {
            "filters": self.filters,
            "biases": self.biases
        }
    
    def set_parameters(self, parameters):
        self.filters = parameters["filters"]
        self.biases = parameters["biases"]


class MaxPool:

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):

        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        # Reducing the feature map size by dividing by the pool size
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        self.output = np.zeros(
            (self.num_channels, self.output_height, self.output_width))

        for channel in range(self.num_channels):
            # Looping through height
            for i in range(self.output_height):
                # Looping through width
                for j in range(self.output_width):

                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = input_data[channel, start_i:end_i, start_j:end_j]
                    # Finding max value from each patch
                    self.output[channel, i, j] = np.max(patch)

        return self.output

    def backward(self, der_out):
        der_input = np.zeros_like(self.input_data)

        for channel in range(self.num_channels):
            # Looping through height
            for i in range(self.output_height):
                # Looping through width
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[channel,
                                            start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)
                    
                    der_input[channel, start_i:end_i,
                              start_j:end_j] = der_out[channel, i, j] * mask

        return der_input


class Fully_Connected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.randn(output_size, 1)

    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)
        log_sum_exp = np.log(sum_exp_values)

        # Compute the softmax probabilities
        probabilities = exp_values / sum_exp_values

        return probabilities

    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)

    def forward(self, input_data):
        self.input_data = input_data
        # Flattening the inputs from the previous layer
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases

        # Applying softmax
        self.output = self.softmax(self.z)
        return self.output

    def backward(self, der_out, learning_rate):
        # Gradient of loss with respect to pre-activation
        der_y = np.dot(self.softmax_derivative(self.output), der_out)
        # Gradient of loss with respect to the weights
        der_w = np.dot(der_y, self.input_data.flatten().reshape(1, -1))

        # Gradient of loss with respect to the biases
        der_b = der_y

        # Gradient of loss with respect to the input data
        der_input = np.dot(self.weights.T, der_y)
        der_input = der_input.reshape(self.input_data.shape)

        # Update weights and biases based on learning rate
        self.weights -= learning_rate * der_w
        self.biases -= learning_rate * der_b

        return der_input
    
    def get_parameters(self):
        return {
            "weights": self.weights,
            "biases": self.biases
        }
    
    def set_parameters(self, parameters):
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]


def cross_entropy_loss(y_true, y_pred):
    num_samples = len(y_true)

    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / num_samples
    return loss


def cross_entropy_loss_gradient(y_true, y_pred):
    num_samples = len(y_true)
    gradient = -y_true / (y_pred + 1e-7) / num_samples
    return gradient

def plot_data(train_loss, train_accuracy, epochs):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    epochs = range(1, epochs + 1)
    ax1.plot(epochs, train_accuracy, label='Training Accuracy', color="#FFA500", marker=".")
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_loss, label='Training Loss', color="#0096FF", marker=".")
    ax2.set_title("Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()


conv = Convolution(data_train[0].shape, 6, 1)
pool = MaxPool(2)
full = Fully_Connected(121, 10)
accuracy_data = []  
loss_data = [] 
epoch_data = []

# Load selected checkpoint
if resume_training and os.path.exists(selected_checkpoint):
    with open(selected_checkpoint, "rb") as file:
        loaded_checkpoint = pickle.load(file)

    conv.set_parameters(loaded_checkpoint["conv"])
    full.set_parameters(loaded_checkpoint["full"])
    accuracy_data = loaded_checkpoint['accuracy_data']
    loss_data = loaded_checkpoint['loss_data']
    start_epoch = loaded_checkpoint["epoch"]
    print(loaded_checkpoint["accuracy_data"], loaded_checkpoint["loss_data"])

with open(selected_checkpoint, 'rb') as file:
    loaded_checkpoint = pickle.load(file)
    plot_data(loaded_checkpoint["loss_data"], loaded_checkpoint["accuracy_data"], loaded_checkpoint["epoch"])

def train_network(X_train, y_train, conv, pool, full, learning_rate=0.01, epochs=24, batch_size=64):
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        # Iterate over batches
        for batch_start in range(0, len(X_train), batch_size):
            batch_end = min(batch_start + batch_size, len(X_train))
            batch_X = X_train[batch_start:batch_end]
            batch_y = y_train[batch_start:batch_end]

            for i in range(len(batch_X)):
                # Forward propagation
                conv_out = conv.forward(X_train[i])
                pool_out = pool.forward(conv_out)
                full_out = full.forward(pool_out)

                # Convert the scalar label to one-hot encoding
                # Assuming there are 10 classes in MNIST
                actual_label_one_hot = np.zeros(10)
                actual_label_one_hot[batch_y[i]] = 1

                loss = cross_entropy_loss(actual_label_one_hot, full_out.flatten())
                total_loss += loss

                # Converting to One-Hot encoding
                one_hot_pred = np.zeros_like(full_out)
                one_hot_pred[np.argmax(full_out)] = 1
                one_hot_pred = one_hot_pred.flatten()

                num_pred = np.argmax(one_hot_pred)
                num_y = np.argmax(batch_y[i])

                if num_pred == num_y:
                    correct_predictions += 1

                # Backward propagation
                gradient = cross_entropy_loss_gradient(
                    actual_label_one_hot, full_out.flatten()).reshape((-1, 1))
                full_back = full.backward(gradient, learning_rate)
                pool_back = pool.backward(full_back)
                conv_back = conv.backward(pool_back, learning_rate)

        # Print epoch statistics
        average_loss = total_loss / len(X_train)
        accuracy = correct_predictions / len(data_train) * 100
        print("Correct predictions: ", correct_predictions)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")
        
         # Print and store results
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Training Loss: {average_loss:.4f} - Training Accuracy: {accuracy:.2f}%")
        
        accuracy_data.append(accuracy)
        loss_data.append(average_loss)
        epoch_data.append(epoch)

        # Save model parameters
        checkpoint_filename = f'{checkpoint_dir}model_parameters_epoch_{epoch + 1}.pkl'
        conv_parameters = conv.get_parameters()
        full_parameters = full.get_parameters()

        with open(checkpoint_filename, "wb") as file:
            pickle.dump({
                "epoch": epoch + 1,
                "conv": conv_parameters, 
                "full": full_parameters,
                "accuracy_data": accuracy_data,
                "loss_data": loss_data,
            }, file)

def predict(input_sample, conv, pool, full):
    # Forward propagation through convolution and pooling
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    # Flattening
    flattened_output = pool_out.flatten()
    # Forward propagation through fully connected layer
    predictions = full.forward(flattened_output)
    return predictions


train_network(data_train, labels_train, conv, pool, full)
