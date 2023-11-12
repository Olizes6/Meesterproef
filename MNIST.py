import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import time

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
        self.biases = np.zeros((num_filters, 1))

    def forward(self, input_data):
        self.input_data = input_data
        num_images = input_data.shape[0]
        output = np.zeros((num_images, *self.output_shape))

        for i in range(num_images):
            for j in range(self.num_filters):
                output[i, j] = correlate2d(
                    self.input_data[i], self.filters[j], mode="valid")

        # ReLU activation function
        output = np.maximum(0, output)

        return output

    def backward(self, der_out, learning_rate):
        num_images = der_out.shape[0]
        der_input = np.zeros_like(self.input_data)
        der_filters = np.zeros_like(self.filters)
        der_biases = (np.sum(der_out, axis=(0, 2, 3), keepdims=True)).reshape(
            self.biases.shape)

        for i in range(num_images):
            for j in range(self.num_filters):
                # Calculating gradient loss
                der_filters[j] = correlate2d(
                    self.input_data[i], der_out[i, j], mode="valid")
                der_input += correlate2d(der_out[i, j],
                                         self.filters[j], mode="full")

        # Updating filters and biases with learning rate
        self.filters -= learning_rate * der_filters / num_images
        self.biases -= learning_rate * der_biases / num_images

        return der_input


class MaxPool:

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input_data = input_data
        self.num_images, self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        self.output = np.zeros(
            (self.num_images, self.num_channels, self.output_height, self.output_width))

        for n in range(self.num_images):
            for channel in range(self.num_channels):
                # Looping through height
                for i in range(self.output_height):
                    # Looping through width
                    for j in range(self.output_width):

                        start_i = i * self.pool_size
                        start_j = j * self.pool_size

                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size

                        patch = input_data[n, channel,
                                           start_i:end_i, start_j:end_j]
                        # Finding max value from each window
                        self.output[n, channel, i, j] = np.max(patch)

        return self.output

    def backward(self, der_out, learning_rate):
        der_input = np.zeros_like(self.input_data)
        for n in range(self.num_images):
            for channel in range(self.num_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size

                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        patch = self.input_data[n, channel,
                                                start_i:end_i, start_j:end_j]

                        mask = patch == np.max(patch)

                        der_input[n, channel, start_i:end_i,
                                  start_j:end_j] = der_out[n, channel, i, j] * mask

        return der_input


def softmax(z):
    shifted_z = z - np.max(z)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z)


def softmax_derivative(s):
    batch_size, num_classes = s.shape
    der_s = np.zeros((batch_size, num_classes, num_classes))
    
    for i in range(batch_size):
        for j in range(num_classes):
            for k in range(num_classes):
                if j == k:
                    der_s[i, j, k] = s[i, j] * (1 - s[i, k])
                else:
                    der_s[i, j, k] = -s[i, j] * s[i, k]
    
    return der_s


class Fully_Connected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size)

    def forward(self, input_data):
        self.input_data = input_data
        flattened_input = input_data.reshape(input_data.shape[0], -1)
        self.z = np.dot(flattened_input, self.weights.T) + self.biases

        # Applying softmax
        self.output = softmax(self.z)
        return self.output

    def backward(self, der_out, learning_rate):
        batch_size = der_out.shape[0]
        
        der_softmax = softmax_derivative(self.output)
        der_z = np.dot(der_out, der_softmax)
        der_w = np.dot(self.input_data.T, der_z) / batch_size

        # Gradient of loss with respect to the biases
        der_b = np.sum(der_z, axis=0) / batch_size

        # Gradient of loss with respect to the input data
        der_input = np.dot(self.weights.T, der_z)

        # Update weights and biases based on learning rate
        self.weights -= learning_rate * der_w
        self.biases -= learning_rate * der_b

        return der_input


def cross_entropy_loss(y_true, y_pred):
    batch_size = y_pred.shape[0]
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / batch_size
    return loss


def cross_entropy_loss_gradient(y_true, y_pred):
    batch_size = y_true.shape[0]
    epsilon = 1e-7
    gradient = -y_true * (1 / (y_pred + epsilon)) / batch_size

    return gradient


def plot_data(loss_value, accuracy_value, epochs):
    plt.plot(epochs, accuracy_value, label="Accuracy")
    plt.plot(epochs, loss_value, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Learning progress")
    plt.show(block=False)
    plt.pause(0.001)


conv = Convolution(data_train[0].shape, 3, 32)
pool = MaxPool(2)
full = Fully_Connected(5408, 10)


def train_network(X_train, y_train, conv, pool, full, learning_rate=0.001, epochs=24, batch_size=64):
    start_time = time.time()
    accuracy_graph = []  # To store accuracy values for each epoch
    loss_graph = []  # To store loss values for each epoch
    epochs_graph = []

    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        # One-hot encode the labels for the entire batch
        batch_labels_one_hot = np.zeros((len(X_train), 10))
        for i, label in enumerate(y_train):
            batch_labels_one_hot[i, label] = 1

        # Iterate over batches
        for batch_start in range(0, len(X_train), batch_size):
            batch_end = min(batch_start + batch_size, len(X_train))
            batch_X = X_train[batch_start:batch_end]
            batch_y = batch_labels_one_hot[batch_start:batch_end]

            # Forward propagation
            conv_out = conv.forward(batch_X)
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)

            # Calculate the cross entropy loss for the entire batch
            batch_loss = cross_entropy_loss(batch_y, full_out)
            print(batch_loss)

            total_loss += batch_loss

            # Calculate the accuracy of the predictions for this batch
            one_hot_pred = softmax(full_out)
            num_pred = np.argmax(one_hot_pred, axis=0)
            num_y = np.argmax(batch_y, axis=1)

            correct_predictions += np.sum(num_pred == num_y)
            # Backward propagation
            gradient = cross_entropy_loss_gradient(batch_y, full_out)
            full_back = full.backward(gradient, learning_rate)
            pool_back = pool.backward(full_back, learning_rate)
            conv_back = conv.backward(pool_back, learning_rate)

            print(
                f"Epoch {epoch + 1}/{epochs}, Batch {batch_start // batch_size + 1}/{num_batches} Loss: {batch_loss:.4f}")

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1} execution time: {epoch_time:.2f} seconds")

        # Calculate the average loss for this epoch
        average_loss = total_loss / num_batches

        # Calculate the accuracy for this epoch
        accuracy = correct_predictions / len(X_train) * 100

        # Append the accuracy and loss values to their respective lists
        accuracy_graph.append(accuracy)
        loss_graph.append(average_loss)
        epochs_graph.append(epoch)

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
