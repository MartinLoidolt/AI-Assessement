import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog

imageSize = 64

def extract_hog_features_multiple(images):
    hog_features = []
    for img in images:
        img = cv2.resize(img, (imageSize, imageSize))
        hog_feat = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
        hog_features.append(hog_feat)
    return np.array(hog_features)

def extract_hog_features_single(img):
    img = cv2.resize(img, (imageSize, imageSize))
    hog_features, hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
    return hog_features, hog_image

def predict_emotion_hog(image_path, model, face_cascade):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    (x, y, w, h) = faces[0]
    face_region = img[y:y+h, x:x+w]

    features, hog_image = extract_hog_features_single(face_region)

    features = features.reshape(1, -1)

    predicted_expression = model.predict(features)[0]

    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    img_with_box = img_rgb.copy()
    cv2.rectangle(img_with_box, (x,y), (x+w,y+h), (12, 163, 111), 1)
    #cv2.putText(img_with_box, predicted_expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 20)

    plt.subplot(1,3,1)
    plt.imshow(img_rgb)
    plt.title(f"Original Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_with_box)
    plt.title(f"Predicted Expression: $\mathbf{{{predicted_expression}}}$")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hog_image, cmap="gray")
    plt.title(f"HOG Features")
    plt.axis("off")

    plt.show()
