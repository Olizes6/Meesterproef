from PIL import Image
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

# load the mnist dataset
mnist = fetch_openml("mnist_784", version=1)



def load_and_resize_image(image_path, target_size): # Load the image and resize it to the target size
    image = Image.open(image_path) 
    image = image.resize(target_size)
    return image

def scale_image(image): # Scale the pixel values to [0, 1]
    scaled_image = np.asarray(image) / 255.0
    return scaled_image

def preprocess_image(image_path, target_size): # Preprocess the image combining the functions above
    image = load_and_resize_image(image_path, target_size)
    preprocessed_image = scale_image(image)
    return preprocessed_image

def display_image(image): # display the image on screen
    plt.imshow(image)
    plt.axis('off')
    plt.show()

image_path = "Uh oh stinky.jpg"
target_size = (255, 255)
preprocessed_image = preprocess_image(image_path, target_size)

display_image(preprocessed_image)