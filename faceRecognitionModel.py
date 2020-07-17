import face_recognition


class FaceRecognitionModel:
    __DEFAULT_FACE_LABEL = "Unknown"

    def __init__(self, known_face_labels, known_face_encodings):
        self.__known_face_labels = known_face_labels
        self.__known_face_encodings = known_face_encodings

    def predictFaceLabel(self, image, face_location, tolerance):
        face_location_list = []
        face_location_list.append(face_location)
        unknown_face_encoding = face_recognition.face_encodings(image, face_location_list)[0]
        return self.__perform1NNPrediction(unknown_face_encoding, tolerance)

    def __perform1NNPrediction(self, unknown_face_encoding, tolerance):
        results = face_recognition.compare_faces(self.__known_face_encodings, unknown_face_encoding,
                                                 tolerance=tolerance)
        indexes = [i for i, x in enumerate(results) if x]
        return self.__known_face_labels[indexes[0]] if len(indexes) >= 1 else self.__DEFAULT_FACE_LABEL
