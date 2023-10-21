import numpy as np
from scipy.signal import correlate2d
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

# load the mnist dataset
mnist = fetch_openml("mnist_784", version=1)

data = mnist.data.astype(np.float32)
labels = mnist.target.astype(np.int32)

class Convolution:

    def __init__(self, input_shape, filter_size, num_filters):
      input_height, input_width = input_shape
      self.num_filters = num_filters
      self.input_shape = input_shape

      # Size of outputs and kernels

      self.filter_shape = (num_filters, filter_size, filter_size) # (3,3)
      self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
      self.filters = np.random.randn(*self.filter_shape)
      self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_data):
        self.input_data = input_data
       
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
          output[i] = correlate2d(self.input_data, self.filters[i], mode="valid")
        #ReLU activation function
        output = np.maximum(0, output)
        return output
    
    def backward(self, der_out, learning_rate):
       der_input = np.zeros_like(self.input_data)
       der_filters = np.zeros_like(self.filters)

       for i in range(self.num_filters):
          #Calculating gradient loss
          der_filters[i] = correlate2d(self.input_data, der_out[i], mode="valid")
          der_input += correlate2d(der_out[i], self.filters[i], mode="full")

       #Updating filters and biases with learning rate
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

               patch = input_data[channel, start_i:end_i, start_j:end_j]
               print(patch)
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
               patch = self.input_data[channel, start_i:end_i, start_j:end_j]

               mask = patch == np.max(patch)
               
               der_input[channel, start_i:end_i, start_j:end_j] = der_out[channel, i, j] * mask

      return der_input

       