import numpy as np
from scipy.signal import correlate2d
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# load the mnist dataset
mnist_data, mnist_labels = fetch_openml(
    "mnist_784", version=1, return_X_y=True, parser="auto")

data = mnist_data.astype(np.float32) / 255
labels = mnist_labels.astype(np.int32)

data = data.values.reshape(-1, 28, 28)

data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

labels_train = labels_train.values


class Convolution:

    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape

        # Size of outputs and kernels

        self.filter_shape = (num_filters, filter_size, filter_size)  # (3,3)
        self.output_shape = (num_filters, input_height -
                             filter_size + 1, input_width - filter_size + 1)

        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

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
            der_input += correlate2d(der_out[i], self.filters[i], mode="full")

        # Updating filters and biases with learning rate
        self.filters -= learning_rate * der_filters
        self.biases -= learning_rate * der_out

        return der_input


class MaxPool:

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):

        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
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
                    # Finding max value from each window
                    self.output[channel, i, j] = np.max(patch)

        return self.output

    def backward(self, der_out, learning_rate):
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


def cross_entropy_loss(predictions, targets):
    num_samples = 10

    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss


def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples

    return gradient


conv = Convolution(data_train[0].shape, 6, 1)
pool = MaxPool(2)
full = Fully_Connected(121, 10)


def train_network(X, y, conv, pool, full, learning_rate=0.001, epochs=200, batch_size=64):
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        # Iterate over batches
        for batch_start in range(0, len(X), batch_size):
            batch_end = min(batch_start + batch_size, len(X))
            batch_X = X[batch_start:batch_end]
            batch_y = y[batch_start:batch_end]

            for i in range(len(batch_X)):
                # Forward propagation
                conv_out = conv.forward(X[i])
                pool_out = pool.forward(conv_out)
                full_out = full.forward(pool_out)

                # Convert the scalar label to one-hot encoding
                # Assuming there are 10 classes in MNIST
                actual_label_one_hot = np.zeros(10)
                actual_label_one_hot[batch_y[i]] = 1

                loss = cross_entropy_loss(full_out.flatten(), y[i])
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
                pool_back = pool.backward(full_back, learning_rate)
                conv_back = conv.backward(pool_back, learning_rate)

            # Print batch statistics (optional)
            average_loss = total_loss / len(batch_X)
            print(
                f"Epoch {epoch + 1}/{epochs}, Batch {batch_start // batch_size + 1}/{len(X) // batch_size}, Loss: {average_loss:.4f}")

        end_time = time.time()
        epoch_time = end_time - start_time
        total_execution_time = epoch_time * epochs
        print(
            f"Estimated total execution time: {total_execution_time} seconds")
        # Print epoch statistics
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(data_train) * 100
        print("Correct predictions: ", correct_predictions)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")


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
