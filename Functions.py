import cv2
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import hog

#Load Faces-------------------------------------------------------------------------------------------------------------
def load_and_detect_faces (folder_path, face_cascade):
  images = []
  labels = []
  for label in os.listdir(folder_path):
    label_path = os.path.join(folder_path, label)
    if os.path.isdir(label_path):
      for img_file in os.listdir(label_path):
        img_path= os.path.join(label_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
          # Detect face
          faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
          for (x, y, w, h) in faces:
            face_region = img[y:y+h, x:x+w] # Crop to face region
            images.append(face_region)
            labels.append(label)
  return images, labels

#optimizes the images by increasing contrast and more
def preprocess_image(img):
  # converting to LAB color space
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l_channel, a, b = cv2.split(lab)

  # Applying CLAHE to L-channel
  # feel free to try different values for the limit and grid size:
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  cl = clahe.apply(l_channel)

  # merge the CLAHE enhanced L-channel with the a and b channel
  limg = cv2.merge((cl, a, b))

  # Converting image from LAB Color model to BGR color spcae
  enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

  return enhanced_img