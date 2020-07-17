import face_recognition
from pathlib import Path
import joblib
import os
import config

known_face_encodings = []
known_face_labels = []
for training_folder in [f.name for f in os.scandir("training_faces") if f.is_dir()]:
    print("Extracting images of", training_folder)
    for training_image in Path("training_faces/" + training_folder).glob(config.IMAGE_FORMAT):
        known_face = face_recognition.load_image_file(training_image)
        face_encoding = face_recognition.face_encodings(known_face)[0]
        known_face_encodings.append(face_encoding)
        known_face_labels.append(training_folder)

joblib.dump(known_face_encodings, "x_train.dat")
joblib.dump(known_face_labels, "y_train.dat")
