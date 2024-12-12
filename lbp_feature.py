import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

from Functions import preprocess_image, preprocess_image

radius = 1
n_points = 8 * radius

def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        img = cv2.resize(img, (64,64))
        lbp = local_binary_pattern(img, n_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.append(hist)
    return np.array(lbp_features)

def predict_emotion_lbp(image_path, model, face_cascade):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected in the image")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (x, y, w, h) = faces[0]
    face_region = img[y:y + h, x:x + w]

    face_region = cv2.resize(face_region, (64, 64))

    lbp = local_binary_pattern(face_region, n_points, radius, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    features = hist.reshape(1, -1)
    predicted_expression = model.predict(features)[0]

    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    img_with_box = img_rgb.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (12, 163, 111), 1)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title(f"Original Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_with_box)
    plt.title(f"Predicted Expression: {predicted_expression}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(lbp, cmap="gray")
    plt.title(f"LBP Features")
    plt.axis("off")

    plt.show()