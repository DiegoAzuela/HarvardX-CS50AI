"""
SOURCE
    https://cs50.harvard.edu/ai/2020/projects/5/traffic/
    
SOLVED BY
    Diego Arnoldo Azuela Rosas

BACKGROUND
    As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.
    In this project, you’ll use TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.
    Several such data sets exist, but for this project, we’ll use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

UNDERSTANDING
    First, take a look at the data set by opening the gtsrb directory. You’ll notice 43 subdirectories in this dataset, numbered 0 through 42. Each numbered subdirectory represents a different category (a different type of road sign). Within each traffic sign’s directory is a collection of images of that type of traffic sign.
    Next, take a look at traffic.py. In the main function, we accept as command-line arguments a directory containing the data and (optionally) a filename to which to save the trained model. The data and corresponding labels are then loaded from the data directory (via the load_data function) and split into training and testing sets. After that, the get_model function is called to obtain a compiled neural network that is then fitted on the training data. The model is then evaluated on the testing data. Finally, if a model filename was provided, the trained model is saved to disk.
    The load_data and get_model functions are left to you to implement.

SPECIFICATION
    Complete the implementation of load_data and get_model in traffic.py.
        The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
            You may assume that data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. Inside each category directory will be some number of image files.
            Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.
            The function should return a tuple (images, labels). images should be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size. labels should be a list of integers, representing the category number for each of the corresponding images in the images list.
            Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the "\" character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
        The get_model function should return a compiled neural network model.
            You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
            The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
            The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
                different numbers of convolutional and pooling layers
                different numbers and sizes of filters for convolutional layers
                different pool sizes for pooling layers
                different numbers and sizes of hidden layers
                dropout
        In a separate file called README.md, document (in at least a paragraph or two) your experimentation process. What did you try? What worked well? What didn’t work well? What did you notice?
    Ultimately, much of this project is about exploring documentation and investigating different options in cv2 and tensorflow and seeing what results you get when you try them!
    You should not modify anything else in traffic.py other than the functions the specification calls for you to implement, though you may write additional functions and/or import other Python standard library modules. You may also import numpy or pandas, if familiar with them, but you should not use any other third-party Python modules. You may modify the global variables defined at the top of the file to test your program with other values.

HINTS
    - Check out the official Tensorflow Keras overview for some guidelines for the syntax of building neural network layers. You may find the lecture source code useful as well.
    - The OpenCV-Python documentation may prove helpful for reading images as arrays and then resizing them.
    - Once you’ve resized an image img, you can verify its dimensions by printing the value of img.shape. If you’ve resized the image correctly, its shape should be (30, 30, 3) (assuming IMG_WIDTH and IMG_HEIGHT are both 30).
    - If you’d like to practice with a smaller data set, you can download a modified dataset that contains only 3 different types of road signs instead of 43.

IMPROVEMENTS
    - 

"""
# SOURCE: 

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # "images" = [list images as numpy.ndarray]. "labels" = [list of integers]
    images, labels = [], []
    # Alert the user of the loading process
    print(f'Loading images from the directory "{data_dir}"')
    # Load the images according to the directory and apply 'IMG_WIDTH' and 'IMG_HEIGHT'
    for filename in os.listdir(os.path.join(data_dir,foldername)):
        img = cv2.imread(path = os.path.join(data_dir,foldername,filename), flag = f)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
        img = img/255   # Normalize for simpler use
        # Append the information
        images.append(img)
        labels.append(int(foldername))
    # Make sure the information is the correct
    if len(images) != len(labels):
        sys.exit("Please verify the information being handled, the data length for 'images' and 'labels' do not match")
    else: 
        print(f'Succesfully uploaded "{len(images)}" with "{len(labels)}" from the dataset')
    return (images,labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    The number of layers and the types of layers you include in between are up to you. You may wish to experiment with: 
        1. different numbers of convolutional and pooling layers
        2. different numbers and sizes of filters for convolutional layers
        3. different pool sizes for pooling layers
        4. different numbers and sizes of hidden layers
        5. dropout
    """
    # Create model with different layers
    model = tf.keras.models.Sequential(
        [
        # Add the different layers for the result following diagram, such that the result is ["Convolutional" > "Pooling" > "Flattening"]
            layers.Dense(units=64,activation="relu",name="Dense-Layer1"),
            layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",input_shape=(IMG_WIDTH,IMG_HEIGHT,3),name="Convolutional-Layer2"),
            layers.MaxPooling2D(data_format=3,pool_size=(3,3),keepdims=False,name="MaxPooling2D-Layer3"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=NUM_CATEGORIES,activation="softmax")
        ]
    )
    # Train Neural Network
    model.compile(
        optimizer = "adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
