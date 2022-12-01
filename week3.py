import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Load the data

# Get current working directory
current_dir = os.getcwd() 

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz") 

# Get only training set
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path) 







from tensorflow.keras.preprocessing.image import load_img
base_dir = "./data/"
happy_dir = os.path.join(base_dir, "happy/")
sad_dir = os.path.join(base_dir, "sad/")
print("Sample happy image:")
plt.imshow(load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
plt.show()
print("\nSample sad image:")
plt.imshow(load_img(f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
plt.show()






 # GRADED FUNCTION: reshape_and_normalize

def reshape_and_normalize(images):
    
    ### START CODE HERE

    # Reshape the images to add an extra dimension
    images = np.reshape(images, (60000, 28, 28, 1))
    
    # Normalize pixel values
    images = np.divide(images, 255)
    
    ### END CODE HERE

    return images
  
  
  
  
  # Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path) 

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")





 # GRADED CLASS: myCallback
### START CODE HERE

# Remember to inherit from the correct class
class myCallback(tf.keras.callbacks.Callback):
    # Define the method that checks the accuracy at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.995):
            print("Reached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True
    #pass
### END CODE HERE






 # GRADED FUNCTION: convolutional_model
def convolutional_model():
    ### START CODE HERE

    # Define the model, it should have 5 layers:
    # - A Conv2D layer with 32 filters, a kernel_size of 3x3, ReLU activation function
    #    and an input shape that matches that of every image in the training set
    # - A MaxPooling2D layer with a pool_size of 2x2
    # - A Flatten layer with no arguments
    # - A Dense layer with 128 units and ReLU activation function
    # - A Dense layer with 10 units and softmax activation function
    model = tf.keras.models.Sequential([ 
      tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ]) 

    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']) 
        
    return model
  
  
  
  
  
  
  
  
   # Save your untrained model
model = convolutional_model()

# Instantiate the callback class
callbacks = myCallback()

# Train your model (this can take up to 5 minutes)
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])







print(f"Your model was trained for {len(history.epoch)} epochs")
