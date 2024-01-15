import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import time
import pickle
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

resume_training = False
start_epoch = 0

checkpoint_dir = 'checkpoints_ontwerpcyclus_3.1/'
selected_checkpoint = (f"{checkpoint_dir}model_parameters_epoch_12.pkl")

# Check if the directory exists; if not, create it
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, der_out, learning_rate):
        der_input = der_out
        for layer in reversed(self.layers):
            der_input = layer.backward(der_input, learning_rate)
        return der_input

class Convolution:

    def __init__(self, input_shape, filter_size, num_filters, momentum):
        if len(input_shape) == 2:
            self.input_height, self.input_width = input_shape
            self.num_channels = 1  # Assuming grayscale images for simplicity
        elif len(input_shape) == 3:
            self.num_channels, self.input_height, self.input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        self.momentum = momentum
        # Size of outputs and kernels

        self.filter_shape = (num_filters, filter_size, filter_size)  # (3,3)
        self.output_shape = (num_filters, self.input_height -
                             filter_size + 1, self.input_width - filter_size + 1)
       
        self.filters = np.random.randn(
            *self.filter_shape) * np.sqrt(2 / np.prod(self.filter_shape[1:]))
        self.biases = np.zeros(self.output_shape)

        self.filter_momentum = np.zeros_like(self.filters)
        self.bias_momentum = np.zeros_like(self.biases)

    def forward(self, input_data):
        self.input_data = input_data
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            if len(self.input_shape) == 2:
                # For the first convolutional layer with a 2D image input
                output[i] = correlate2d(self.input_data, self.filters[i, :, :], mode="valid")
            else:
                # For subsequent convolutional layers with multiple feature maps
                for c in range(self.num_channels):
                    output[i] += correlate2d(self.input_data[c, :, :], self.filters[i, :, :], mode="valid")

            # Add bias and apply ReLU activation function
            output[i] += self.biases[i]
            output[i] = np.maximum(0, output[i])

        return output

    def backward(self, der_out, learning_rate):
        der_input = np.zeros_like(self.input_data)
        der_filters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
            if len(self.input_shape) == 2:
                # For the first convolutional layer with a 2D image input
                der_filters[i] = correlate2d(self.input_data, der_out[i], mode="valid")
                der_input += correlate2d(der_out[i], self.filters[i, :, :], mode="full")
            else:
                # For subsequent convolutional layers with multiple feature maps
                for c in range(self.num_channels):
                    der_filters[i] += correlate2d(self.input_data[c, :, :], der_out[i], mode="valid")
                    der_input[c] += correlate2d(der_out[i], self.filters[i, :, :], mode="full")
            
        self.filter_momentum = self.momentum * self.filter_momentum + learning_rate * der_filters
        self.bias_momentum = self.momentum * self.bias_momentum + learning_rate * der_out

        # Updating filters and biases with learning rate
        self.filters -= self.filter_momentum
        self.biases -= self.bias_momentum
        return der_input
    
    def get_parameters(self):
        return {
            "filters": self.filters,
            "biases": self.biases,
            "filter_momentum": self.filter_momentum,
            "bias_momentum": self.bias_momentum
        }
    
    def set_parameters(self, parameters):
        self.filters = parameters["filters"]
        self.biases = parameters["biases"]
        self.filter_momentum = parameters["filter_momentum"]
        self.bias_momentum = parameters["bias_momentum"]

class MaxPool:

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        self.output_shape = (self.num_channels, self.output_height, self.output_width)

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


def softmax(z):
    shifted_z = z - np.max(z)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=0)


def softmax_derivative(s):
    return np.diagflat(s) - np.dot(s, s.T)


class Fully_Connected:

    def __init__(self, input_size, output_size, momentum, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros(output_size).reshape(-1, 1)
        self.momentum = momentum
        self.activation = activation

        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

    def forward(self, input_data):
        self.input_data = input_data
        self.flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, self.flattened_input.T) + self.biases
        
        if self.activation == "softmax":
            # Applying softmax
            self.output = softmax(self.z)
        elif self.activation == "relu":
            self.output = np.maximum(0, self.z)
        return self.output

    def backward(self, der_loss, learning_rate):
        if self.activation == "softmax":
            der_y = np.dot(softmax_derivative(self.output), der_loss)
        elif self.activation == "relu":
            der_y = (self.output > 0) * der_loss

        # Gradient of loss with respect to the input data
        der_input = np.dot(self.weights.T, der_y).reshape(self.input_data.shape)

        der_w = np.dot(der_y, self.flattened_input)
        # Gradient of loss with respect to the biases
        der_b = der_y

        self.weight_momentum = self.momentum * self.weight_momentum + learning_rate * der_w
        self.bias_momentum = self.momentum * self.bias_momentum + learning_rate * der_b

        # Update weights and biases based on learning rate
        self.weights -= self.weight_momentum
        self.biases -= self.bias_momentum

        return der_input
    
    def get_parameters(self):
        return {
            "weights": self.weights,
            "biases": self.biases,
            "weight_momentum": self.weight_momentum,
            "bias_momentum": self.bias_momentum
        }
    
    def set_parameters(self, parameters):
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]
        self.weight_momentum = parameters["weight_momentum"]
        self.bias_momentum = parameters["bias_momentum"]


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

def plot_data(train_loss, train_accuracy, val_loss, val_accuracy, epochs):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    epochs = range(1, epochs + 1)
    ax1.plot(epochs, train_accuracy, label='Training Accuracy', color="#FFA500", marker=".")
    ax1.plot(epochs, val_accuracy, label='Validation Accuracy', color="#FF0000", marker=".")
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_loss, label='Training Loss', color="#0096FF", marker=".")
    ax2.plot(epochs, val_loss, label='Validation Loss', color="#FF0000", marker=".")
    ax2.set_title("Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.show()

#conv = Convolution(data_train[0].shape, 3, 32, 0)
#pool = MaxPool(2)
#full = Fully_Connected(5408, 10, 0)
    
model = Sequential()

model.add(Convolution(input_shape=data_train[0].shape, filter_size=3, num_filters=32, momentum=0.9))
model.add(MaxPool(2))
model.add(Convolution(input_shape=(32, 13, 13), filter_size=3, num_filters=64, momentum=0.9))
model.add(MaxPool(2))

num_features = np.prod([64, 5, 5])
model.add(Fully_Connected(input_size=num_features, output_size=128, momentum=0.9, activation="relu"))

model.add(Fully_Connected(input_size=128, output_size=10, momentum=0.9, activation="softmax"))

accuracy_data = []  
loss_data = [] 
val_accuracy_data = []
val_loss_data = []
epoch_data = []

def save_model(model, filename):
    model_params = {}
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'get_parameters'):
            model_params[f"layer_{idx}"] = layer.get_parameters()
    return model_params

def load_model(model, filename):
    with open(filename, 'rb') as file:
        model_params = pickle.load(file)
        
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'set_parameters'):
            layer.set_parameters(model_params[f"layer_{idx}"])
if resume_training and os.path.exists(selected_checkpoint):
    load_model(model, selected_checkpoint)

# Load selected checkpoint
# if resume_training and os.path.exists(selected_checkpoint):
#     with open(selected_checkpoint, "rb") as file:
#         loaded_checkpoint = pickle.load(file)

#     conv.set_parameters(loaded_checkpoint["conv"])
#     full.set_parameters(loaded_checkpoint["full"])
#     accuracy_data = loaded_checkpoint['accuracy_data']
#     loss_data = loaded_checkpoint['loss_data']
#     val_accuracy_data = loaded_checkpoint['val_accuracy_data']
#     val_loss_data = loaded_checkpoint['val_loss_data']
#     start_epoch = loaded_checkpoint["epoch"]
#     print(loaded_checkpoint)

# with open(selected_checkpoint, 'rb') as file:
#     loaded_checkpoint = pickle.load(file)
#     print(loaded_checkpoint["loss_data"], loaded_checkpoint["accuracy_data"], loaded_checkpoint["val_loss_data"], loaded_checkpoint["val_accuracy_data"], loaded_checkpoint["epoch"])
#     plot_data(loaded_checkpoint["loss_data"], loaded_checkpoint["accuracy_data"], loaded_checkpoint["val_loss_data"], loaded_checkpoint["val_accuracy_data"], loaded_checkpoint["epoch"])
    
def train_network(X_train, y_train, X_val, y_val, model, learning_rate=0.01, epochs=24):
    start_time = time.time()
    initial_learning_rate = learning_rate
    minimum_learning_rate = 1e-5

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        correct_predictions = 0

        # Reduce learning rate to reach convergence faster
        new_learning_rate = max(minimum_learning_rate, ((1 / (1 + 1 * epoch)) * initial_learning_rate))
        learning_rate = new_learning_rate 
        print("current learning rate: ", new_learning_rate)

        X_train, y_train = shuffle(X_train, y_train)

        for i in range(len(X_train)):

            # Forward propagation
            model_output = model.forward(X_train[i])

            # Convert the scalar label to one-hot encoding
            actual_label_one_hot = np.zeros(10)
            actual_label_one_hot[y_train[i]] = 1

            loss = cross_entropy_loss(actual_label_one_hot, model_output.flatten())
            total_loss += loss

            one_hot_pred = np.zeros_like(model_output.flatten())
            one_hot_pred[np.argmax(model_output)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            correct_predictions += np.sum(num_pred == y_train[i])

            if (i + 1) % 10 == 0:
                print(f"Accuracy: {((correct_predictions / (i + 1)) * 100):.2f}% Loss: {total_loss / (i + 1)}")
            # Backward propagation
            gradient = cross_entropy_loss_gradient(actual_label_one_hot, model_output.flatten()).reshape((-1, 1))
            model.backward(gradient, learning_rate)

        # Calculate the average training loss for this epoch
        average_loss = total_loss / len(X_train)

        # Calculate the training accuracy for this epoch
        accuracy = correct_predictions / len(X_train) * 100

        val_total_loss = 0.0
        val_correct_predictions = 0

        for i in range(len(X_val)):
            model_output = model.forward(X_val[i])

            # Convert the scalar label to one-hot encoding
            actual_label_one_hot = np.zeros(10)
            actual_label_one_hot[y_val[i]] = 1

            loss = cross_entropy_loss(actual_label_one_hot, model_output.flatten())
            val_total_loss += loss

            one_hot_pred = np.zeros_like(model_output.flatten())
            one_hot_pred[np.argmax(model_output)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            val_correct_predictions += np.sum(num_pred == y_val[i])

        # Calculate validation set accuracy and loss
        val_average_loss = val_total_loss / len(X_val)
        val_accuracy = val_correct_predictions / len(X_val) * 100

        # Print and store results
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Training Loss: {average_loss:.4f} - Training Accuracy: {accuracy:.2f}% - "
              f"Validation Loss: {val_average_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")
        
        accuracy_data.append(accuracy)
        loss_data.append(average_loss)
        val_accuracy_data.append(val_accuracy)
        val_loss_data.append(val_average_loss)
        epoch_data.append(epoch)

        # Save model parameters
        checkpoint_filename = f'{checkpoint_dir}model_parameters_epoch_{epoch + 1}.pkl'
        #conv_parameters = conv.get_parameters()
        #full_parameters = full.get_parameters()

        model_params = save_model(model, checkpoint_filename)

        with open(checkpoint_filename, "wb") as file:
            pickle.dump({
                "epoch": epoch + 1,
                "model": model_params,
                "accuracy_data": accuracy_data,
                "loss_data": loss_data,
                "val_accuracy_data": val_accuracy_data,
                "val_loss_data": val_loss_data,
            }, file)
            
    end_time = time.time()
    total_execution_time = end_time - start_time
    print(f"Total execution time: {total_execution_time:.2f} seconds")

    # Print final epoch statistics
    print(f"Final Training Accuracy: {accuracy:.2f}%, Final Training Loss: {average_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.2f}%, Final Validation Loss: {val_average_loss:.4f}")

    # Call this function after training is complete
    plot_data(loss_data, accuracy_data, val_loss_data, val_accuracy_data, epoch_data)

train_network(data_train, labels_train, data_test, labels_test, model)
