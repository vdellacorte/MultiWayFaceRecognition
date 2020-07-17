import cv2
import face_recognition
import os
from cv2 import CascadeClassifier


class FaceDetector:
    def __init__(self, model):
        self.__model = model

    def detectBBoxes(self, image, scale_factor=1.1, min_neighbors=5, face_min_size=(30, 30),
                     number_times_to_upsample=1):
        if self.__model == 'vj':
            return self.__getCv2BBoxes(image, scale_factor, min_neighbors, face_min_size)
        else:
            return self.__getAdamBBoxes(image, self.__model, number_times_to_upsample)

    def __getCv2BBoxes(self, image, scale_factor, min_neighbors, face_min_size):
        detector = self.__getDetector()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bboxes = detector.detectMultiScale(gray, scaleFactor=scale_factor,
                                           minNeighbors=min_neighbors,
                                           minSize=face_min_size)

        face_locations = [(int(y), int(x + w), int(y + h), int(x)) for (x, y, w, h) in bboxes]
        return face_locations

    def __getDetector(self):
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
        detector = CascadeClassifier(haar_model)
        return detector

    def __getAdamBBoxes(self, image, model, number_times_to_upsample):
        face_locations = face_recognition.face_locations(image, model=model,
                                                         number_of_times_to_upsample=
                                                         number_times_to_upsample)
        return face_locations

