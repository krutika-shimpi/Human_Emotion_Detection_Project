# Define a function to plot the loss curves
import matplotlib.pyplot as plt
import numpy as np
import json

def save_model(model, filepath):
  """
  Saves the model by pickling the model configuration and weights separately.

  Args:
    model: The model to save.
    filepath: The path to save the model.
  """
  # Save the model configuration as JSON
  with open(filepath + '.json', 'w') as f:
    f.write(model.to_json())

  # Save the model weights, ensuring the filename ends with '.weights.h5'
  model.save_weights(filepath + '.weights.h5') # Changed '.h5' to '.weights.h5'

def load_model(filepath):
  """
  Loads the model from the saved configuration and weights.

  Args:
    filepath: The path to the saved model.

  Returns:
    The loaded model.
  """
  # Load the model configuration from JSON
  with open(filepath + '.json', 'r') as f:
    model_config = f.read()

  # Create a new model instance from the configuration
  model = tf.keras.models.model_from_json(model_config)

  # Load the model weights, ensuring the filename ends with '.weights.h5'
  model.load_weights(filepath + '.weights.h5') # Changed '.h5' to '.weights.h5'

  return model




def plot_loss_curves(history, regression = None, classification = None, val_data = None):
  """
  This function plots the loss curves by accepting an input parameter history.
  """

  plt.figure(figsize = (12,6))
  
  ### CLASSIFICATION
  if classification:
    loss = history.history['loss']
    accuracy = history.history['accuracy']
    epochs = range(len(history.history['loss']))
    
    if val_data:
      val_loss = history.history['val_loss']
      val_accuracy = history.history['val_accuracy']
    
      # Plot the loss and accuracy curves
      plt.subplot(1, 2, 1)
      plt.plot(epochs, loss, label = 'Training loss')
      plt.plot(epochs, val_loss, label = 'Validation loss')
      plt.title('Training Vs. Validation loss curves')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
    
      plt.subplot(1, 2, 2)
      plt.plot(epochs, accuracy, label = 'Training accuracy')
      plt.plot(epochs, val_accuracy, label = 'Validation accuracy')
      plt.title('Training Vs. Validation accuracy curves')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.legend()

    else:
      # Find the losses and accuracies
      
      plt.subplot(1, 2, 1)
      plt.plot(epochs, loss, label = 'Loss')
      plt.title('Loss curves')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      
      plt.subplot(1, 2, 2)
      plt.plot(epochs, accuracy, label = 'Accuracy')
      plt.title('Accuracy')
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.legend()

  ### REGRESSION        
  elif regression:
    # Find the losses and metric
    loss = history.history['loss']    
    mae = history.history['mae']    
    epochs = range(len(history.history['loss']))

    if val_data:
      val_loss = history.history['val_loss']
      val_mae = history.history['val_mae']

      plt.subplot(1, 2, 1)
      plt.plot(epochs, loss, label = 'Training loss')
      plt.plot(epochs, val_loss, label = 'Validation loss')
      plt.title('Training Vs. Validation loss curves')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(epochs, mae, label = 'Training error')
      plt.plot(epochs, val_mae, label = 'Validation error')
      plt.title('Training Vs. Validation error')
      plt.xlabel('Epochs')
      plt.ylabel('Mean Absolute Error')
      plt.legend()
    
    else:
      plt.subplot(1, 2, 1)
      plt.plot(epochs, loss, label = 'Loss')
      plt.title('Loss curves')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(epochs, mae, label = 'Error')
      plt.title('Error')
      plt.xlabel('Epochs')
      plt.ylabel('Mean Absolute Error')
      plt.legend()   

  plt.tight_layout()
  plt.show()

# Let's functionize making prediction
import numpy
def make_predictions(model, dataset):
  """
    Generates predictions for a given dataset using the specified model.

    Args:
        model (tf.keras.Model or similar): The trained model used for making predictions.
        dataset (tf.data.Dataset or iterable): A dataset containing image-label pairs.

    Returns:
        tuple:
            - y_true (numpy.ndarray): Flattened array of true labels.
            - y_pred (numpy.ndarray): Flattened array of predicted labels.

    Note:
        Ensure the dataset provides labels in one-hot encoded format,
        as `np.argmax` is used to convert them into class indices.
    """

  labels = []
  predictions = []

  for image, label in dataset:
    labels.append(label)
    predictions.append(model(image))

  y_true = np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()])
  y_pred = np.concatenate([np.argmax(predictions[:-1], axis = -1).flatten(), np.argmax(predictions[-1], axis = -1).flatten()])

  return y_true, y_pred


# let's create a helper function to compare different metrics across different models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def calculate_results(y_true, y_preds):
  """
  Returns a dictionary of all the metrics needed to compare the model.

  Args:
    y_true:
      Ground truth labels or actual labels

    y_pred:
      Labels predicted by the model.
  """
  # Let's calculate the metrics
  accuracy = accuracy_score(y_true, y_preds)
  precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_preds, average = 'weighted')

  metrics = {
      "accuracy" : accuracy,
      'precision' : precision,
      'recall': recall,
      'f1_score' : fscore
  }

  return metrics

  
import pathlib 
import os 
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


# Create a function to make predictions on multiple images picked at random
def make_prediction_random_images(data_dir, model, CLASS_NAMES):

  """
  This function makes the predictions on randomly picked images from the given
  directory and plots them against the truth labels.

    Args:
      - data_dir : it takes the path of the data
      - model : the model for making predictions
  """

  # Extract paths from the data directory
  paths = [dirnames for (dirnames, _, _) in os.walk(data_dir)][1:]

  # Setup figure size
  plt.figure(figsize = (12, 6))

  for i in range(6):
    # Plot 6 random images and make prediction on it using the model
    plt.subplot(2, 3, i + 1)

    # Randomly select a path from paths
    random_path = random.choice(paths)

    # Choose an image randomly from the random path
    random_image = random.choice(os.listdir(random_path))
    random_image_path = random_path + '/' + random_image

    # Read image
    read_image = cv2.imread(random_image_path)

    # Resize image to expected size by the model
    read_image = cv2.resize(read_image, (256, 256)) # Resizing to (256,256)

    # Turn the image into tensor
    image_tensor = tf.constant(read_image, dtype = tf.float32)

    # Fetch the actual label from the image
    actual_label = random_path.split('/')[-1]

    # Add an extra dimension to the image
    image_tensor_dim = tf.expand_dims(image_tensor, axis = 0)

    # Make prediction on the image
    pred_probs = model.predict(image_tensor_dim)

    # Fetch predicted class label
    pred_class_label = CLASS_NAMES[tf.argmax(pred_probs[0], axis = 0).numpy()]

    # Display the images
    plt.imshow(image_tensor/255.)
    plt.title(f'Actual Label : {actual_label}\nPredicted Label : {pred_class_label}\nPrediction Probabilty: {np.round(max(pred_probs[0]) * 100, 2)}%')

  plt.tight_layout()

  return None

# Create a function to make predictions on multiple images picked at random
def make_prediction_feature_maps(data_dir, model):

  """
  This function makes the predictions on randomly picked images from the given
  directory and returns the feature map

    Args:
      - data_dir : it takes the path of the data
      - model : the model for making predictions
  """

  # Extract paths from the data directory
  paths = [dirnames for (dirnames, _, _) in os.walk(data_dir)][1:]

  # Randomly select a path from paths
  random_path = random.choice(paths)

  # Choose an image randomly from the random path
  random_image = random.choice(os.listdir(random_path))
  random_image_path = random_path + '/' + random_image

  # Read image
  read_image = cv2.imread(random_image_path)

  # Resize image to expected size by the model
  read_image = cv2.resize(read_image, (256,256)) # Resizing to (256,256)

  # Turn the image into tensor
  image_tensor = tf.constant(read_image, dtype = tf.float32)

  # Fetch the actual label from the image
  actual_label = random_path.split('/')[-1]

  # Add an extra dimension to the image
  image_tensor_dim = tf.expand_dims(image_tensor, axis = 0)

  # Make prediction on the image
  feature_maps = model.predict(image_tensor_dim)

  return feature_maps

def plot_random_images(model, images, true_labels, classes):
  """
  It plots the random images with their predicted labels and actual labels.

  Args:
    model: It takes an input model for making predictions.
    images: The data from which we can pick random images
    true_labels: The actual label for the images to compare with.
    classes: To output the respective class for each label.
  
  Returns:
    "Plots the random images picked from the data along with the actual and predicted labels."
  """
  # Make predictions on the data and convert into its labels
  pred_probs = model.predict(images)
  pred_labels = pred_probs.argmax(axis = 1)

  plt.figure(figsize = (12, 9))
  # Plot 6 random images from the data with their labels
  for i in range(6):
    plt.subplot(2, 3, i + 1)
    random_idx = np.random.randint(len(images)-1)
    target_image = images[random_idx]
    target_label = classes[pred_labels[random_idx]]
    true_label = classes[true_labels[random_idx]]

    # Plot the labels in green if the predictions are correct
    if target_label == true_label:
      color = 'green'
    else:
      color = 'red'

    # Plot the image
    plt.imshow(target_image, extent=[0, 0.4, 0, 0.35], cmap = plt.cm.binary)

    # Adding the xlabel information
    plt.xlabel(f'Actual Label: {true_label}\nPredicted Label: {target_label}\nconfidence: {np.round(tf.reduce_max(pred_probs[random_idx]), 1) * 100}%',
               color = color)
    plt.tight_layout()
    
import itertools
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.twilight) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes),
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)  
 
 
import matplotlib.image as mpimg
import os
import pathlib
 # Plot some random images from the data 
def plot_random_images(data_dir):
  """
  Returns a plot of randomly picked images from the data.
  """
  # Pick a path from paths and a random image
  paths = [dir for (dir,_,_) in os.walk(data_dir)]
  paths = paths[1:]

  plt.figure(figsize = (12, 6))
  for i in range(6):
    plt.subplot(2, 3, i + 1)
    random_path = np.random.choice(paths)
    random_image = np.random.choice(os.listdir(random_path), 1)

    # Read the image
    img = mpimg.imread(random_path + '/' + random_image[0].decode())
    # OR
    # os.path.join(random_path, random_image[0].decode())

    # Plot the image
    plt.imshow(img)
    plt.title(f"{random_path.split('/')[1]}")
    plt.xlabel(f"Image shape: {img.shape}")
    
  plt.tight_layout()
  plt.show()

  return img
