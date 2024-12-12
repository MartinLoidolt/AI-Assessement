from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Functions import *
import hog_feature as hog
import lbp_feature as lbp
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#USER INPUTS-------------------------------------------------------------
datasets = ["JAFFE", "MMAFEDB", "CK_dataset"]
featureExtractionMethods = ["LBP", "HOG"]
classifiers = ["DecisionTree", "SVM"]
testDatasets = ["standard", "own"]

string_dataset_options = ""
for i, dataset in enumerate(datasets):
    string_dataset_options += " " + str(i + 1) + " " + dataset + " |"

userInput = input("Choose Dataset |" + string_dataset_options + ": ")
chosenDatasetIndex = int(userInput) - 1
print("Chosen Dataset: " + datasets[chosenDatasetIndex])

string_classifier_options = ""
for i, classifier in enumerate(classifiers):
    string_classifier_options += " " + str(i + 1) + " " + classifier + " |"

userInput = input("Choose Classifier |" + string_classifier_options + ": ")
chosenClassifierIndex = int(userInput) - 1
print("Chosen Classifier: " + classifiers[chosenClassifierIndex])

string_feature_options = ""
for i, featureExtractionMethod in enumerate(featureExtractionMethods):
    string_feature_options += " " + str(i + 1) + " " + featureExtractionMethod + " |"

userInput = input("Choose Feature Extraction Method |" + string_feature_options + ": ")
chosenFeatureIndex = int(userInput) - 1
print("Chosen Method: " + featureExtractionMethods[chosenFeatureIndex])

string_Classifier_options = ""
for i, testDataset in enumerate(testDatasets):
    string_Classifier_options += " " + str(i + 1) + " " + testDataset + " |"

userInput = input("Choose Test Dataset |" + string_Classifier_options + ": ")
chosenTestDatasetIndex = int(userInput) - 1
print("Chosen Test Dataset: " + testDatasets[chosenTestDatasetIndex])

#TRAINING------------------------------------------------------------------
train_folder_path = datasets[chosenDatasetIndex] + "/train"

test_folder_path = None

if testDatasets[chosenTestDatasetIndex] == "standard":
    test_folder_path = datasets[chosenDatasetIndex] + "/test"
elif testDatasets[chosenTestDatasetIndex] == "own":
    test_folder_path = "testimages"

train_images, train_labels = load_and_detect_faces(train_folder_path, face_cascade)
test_images, test_labels = load_and_detect_faces(test_folder_path, face_cascade)
train_features = None
test_features = None

if featureExtractionMethods[chosenFeatureIndex] == "LBP":
    train_features = lbp.extract_lbp_features(train_images)
    test_features = lbp.extract_lbp_features(test_images)
elif featureExtractionMethods[chosenFeatureIndex] == "HOG":
    train_features = hog.extract_hog_features_multiple(train_images)
    test_features = hog.extract_hog_features_multiple(test_images)

classifier = None

if classifiers[chosenClassifierIndex] == "DecisionTree":
    classifier = DecisionTreeClassifier(random_state=10) #40
    classifier.fit(train_features, train_labels)
elif classifiers[chosenClassifierIndex] == "SVM":
    classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=7, gamma="scale"))
    classifier.fit(train_features, train_labels)

y_pred = classifier.predict(test_features)

accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix: ")
print(conf_matrix)

while True :
    image_path = "./testimages/" + input("Enter Filename in /Testimages: ")

    try:
        if featureExtractionMethods[chosenFeatureIndex] == "LBP":
            lbp.predict_emotion_lbp(image_path, classifier, face_cascade)
        elif featureExtractionMethods[chosenFeatureIndex] == "HOG":
            hog.predict_emotion_hog(image_path, classifier, face_cascade)
    except:
        print("Something went wrong try again:")

