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

data_train = data_train / 255.0
data_train = data_train.reshape(-1, 28, 28)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols=1)

resume_training = True
start_epoch = 0

checkpoint_dir = 'checkpoints/'
selected_checkpoint = "checkpoints/model_parameters_epoch_2.pkl"

# Check if the directory exists; if not, create it
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

class Convolution:

    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape

        # Size of outputs and kernels

        self.filter_shape = (num_filters, filter_size, filter_size)  # (3,3)
        self.output_shape = (num_filters, input_height -
                             filter_size + 1, input_width - filter_size + 1)
        self.filters = np.random.randn(
            *self.filter_shape) * np.sqrt(2 / (input_height * input_width * 1))  # 1 for the amount of channels because MNIST images are greyscale images
        self.biases = np.zeros(self.output_shape)

    def forward(self, input_data):
        self.input_data = input_data
        output = np.zeros(self.output_shape)

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
            der_input += correlate2d(der_out[i],
                                        self.filters[i], mode="full")

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
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        for channel in range(self.num_channels):
            # Looping through height
            for i in range(self.output_height):
                # Looping through width
                for j in range(self.output_width):

                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = input_data[channel,
                                        start_i:end_i, start_j:end_j]
                    # Finding max value from each window
                    self.output[channel, i, j] = np.max(patch)

        return self.output

    def backward(self, der_out):
        der_input = np.zeros_like(self.input_data)

        for channel in range(self.num_channels):
            for i in range(self.output_height):
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


def softmax(z):
    shifted_z = z - np.max(z)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=0)


def softmax_derivative(s):
    return np.diagflat(s) - np.dot(s, s.T)


class Fully_Connected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input_data):
        self.input_data = input_data
        self.flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, self.flattened_input.T) + self.biases

        # Applying softmax
        self.output = softmax(self.z)
        return self.output

    def backward(self, der_loss, learning_rate):
        der_y = np.dot(softmax_derivative(self.output), der_loss)
        # Gradient of loss with respect to the input data
        der_input = np.dot(self.weights.T, der_y).reshape(self.input_data.shape)

        der_w = np.dot(der_y, self.flattened_input)
        # Gradient of loss with respect to the biases
        der_b = der_y

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
    num_samples = y_true.shape[0]

    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / num_samples
    return loss


def cross_entropy_loss_gradient(y_true, y_pred):
    num_samples = y_true.shape[0]
    epsilon = 1e-7
    gradient = -y_true / (y_pred + epsilon) / num_samples

    return gradient

def plot_data(loss, accuracy, epochs):
   print(epochs, accuracy, loss)
   ax1.plot(epochs, accuracy, color="#FFA500")
   ax2.plot(epochs, loss, color="#0096FF")

   ax1.set_title("Model Accuracy")
   ax1.set_ylabel("Accuracy")
   ax1.set_xlabel("Epoch")

   ax2.set_title("Model Loss")
   ax2.set_ylabel("Loss")
   ax2.set_xlabel("Epoch")

   plt.tight_layout()
   plt.show()


conv = Convolution(data_train[0].shape, 3, 32)
pool = MaxPool(2)
full = Fully_Connected(5408, 10)

if resume_training and os.path.exists(selected_checkpoint):
    with open(selected_checkpoint, "rb") as file:
        loaded_checkpoint = pickle.load(file)

    conv.set_parameters(loaded_checkpoint["conv"])
    full.set_parameters(loaded_checkpoint["full"])
    start_epoch = loaded_checkpoint["epoch"]
    print("Loaded checkpoint: ", loaded_checkpoint)

def train_network(X_train, y_train, conv, pool, full, learning_rate=0.01, epochs=24):
    start_time = time.time()
    accuracy_graph = []  # To store accuracy values for each epoch
    loss_graph = []  # To store loss values for each epoch
    initial_learning_rate = learning_rate
    epoch_graph = []
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        correct_predictions = 0
        # Reduce learning rate to reach convergence faster
        new_learning_rate = (1 / (1 + 1.5 * epoch)) * initial_learning_rate
        learning_rate = new_learning_rate 
        print("current learning rate: ", new_learning_rate)
        for i in range(len(X_train)):

            # Forward propagation
            conv_out = conv.forward(X_train[i])
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)

            # Convert the scalar label to one-hot encoding
            # Assuming there are 10 classes in MNIST
            actual_label_one_hot = np.zeros(10)
            actual_label_one_hot[y_train[i]] = 1

            loss = cross_entropy_loss(actual_label_one_hot, full_out.flatten())
            total_loss += loss

            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            # Calculate the accuracy of the predictions for this batch
            num_pred = np.argmax(one_hot_pred)
           
            correct_predictions += np.sum(num_pred == y_train[i])

            if (i + 1) % 500 == 0:
                print(f"Accuracy: {((correct_predictions / (i + 1)) * 100):.2f}% Loss: {total_loss / (i + 1)}")
            # Backward propagation
            gradient = cross_entropy_loss_gradient(actual_label_one_hot, full_out.flatten()).reshape((-1, 1))
            full_back = full.backward(gradient, learning_rate)
            pool_back = pool.backward(full_back)
            conv_back = conv.backward(pool_back, learning_rate)
        end_time = time.time()
        epoch_time = end_time - start_time

        # Calculate the average loss for this epoch
        average_loss = total_loss / len(X_train)

        # Calculate the accuracy for this epoch
        accuracy = correct_predictions / len(X_train) * 100
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}% execution time: {epoch_time:.2f} seconds")
        # Append the accuracy and loss values to their respective lists
        accuracy_graph.append(accuracy)
        loss_graph.append(average_loss)
        epoch_graph.append(epoch)
        #plot_data(loss_graph, accuracy_graph, epoch_graph)

        # Save model parameters
        checkpoint_filename = f'{checkpoint_dir}model_parameters_epoch_{epoch + 1}.pkl'
        conv_parameters = conv.get_parameters()
        full_parameters = full.get_parameters()

        with open(checkpoint_filename, "wb") as file:
            pickle.dump({
                "epoch": epoch + 1,
                "conv": conv_parameters, 
                "full": full_parameters
            }, file)
            
    end_time = time.time()
    total_execution_time = end_time - start_time
    print(f"Total execution time: {total_execution_time:.2f} seconds")

    # Print final epoch statistics
    print(f"Final Accuracy: {accuracy:.2f}%, Final Loss: {average_loss:.4f}")


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

predictions = []

for data in data_test:
    pred = predict(data, conv, pool, full)
    one_hot_pred = np.zeros_like(pred)
    one_hot_pred[np.argmax(pred)] = 1
    predictions.append(one_hot_pred.flatten())

predictions = np.array(predictions)

print(predictions)