### Ageing_Sign_Detect

from keras.models import model_from_json # loading the model from json
from collections import OrderedDict
from imutils import face_utils
import imutils
import cv2
import dlib
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to model.json file")
ap.add_argument("-w", "--weight", required=True, help="patht to weights.h5 file")
ap.add_argument("-c", "--cascade", required=True, help="path to cascade.xml file")
ap.add_argument("-i", "--image", required=True, help="path to image")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to shape-predictor.dat file")
args = vars(ap.parse_args())

#loading the model and weights
json_file = open(args["model"], "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(args["weight"])
print("Model Loaded Successfully!!")



def visualize_facial_landmarks(image, shape, colors=None, alpha=0.6):

  FACIAL_LANDMARKS_IDXS = OrderedDict([("left_eye", (42, 48)),
                                       ("right_eye", (36, 42))])

  overlay = image.copy()
  output = image.copy()

  if colors is None:
    colors = (19, 199, 109)

  for  (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):

    (j, k) = FACIAL_LANDMARKS_IDXS[name]
    pts = shape[j:k]
    
    for index, element in enumerate(pts):
      pts[index] = [element[0], element[1]+8]

    hull = cv2.convexHull(pts)
    cv2.drawContours(overlay, [hull], -1, colors, -1)
    
  cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)

  return output

def puffy_localise(image):
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape-predictor"])
    
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
      shape = predictor(gray, rect)
      shape = face_utils.shape_to_np(shape)

    output = visualize_facial_landmarks(image, shape)
    return output

# Testing the model against random images taken from the internet
image = cv2.imread(args["image"])
try:
    image = imutils.resize(image, width=500)
except:
    print("Incorrect Path!! Please provid correct path")

face_cascade = cv2.CascadeClassifier(args["cascade"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.05, 5)

labels = np.array(['dark', 'notOld', 'puffy', 'wrinkles'], dtype=object)

for (x, y, w, h) in faces:
  detectedFace = image[x:x+w, y:y+h]
  detectedFace = cv2.resize(detectedFace, (244, 244))
  detectedFace = detectedFace.reshape(1, 244, 244, 3)
  predictions=loaded_model.predict(detectedFace)[0]
  cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
  
  count = 1
  for i in predictions:
    classname = labels[np.where(predictions == i)]
    accuracy = i*100
    label = "{}: {:.2f}%".format(classname, accuracy)
    
    if classname == 'puffy' and accuracy > 75:
      image = puffy_localise(image)
    
    cv2.putText(image, label, (10, count*30+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    count += 1
  print(predictions)

  cv2.imshow("Title", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

