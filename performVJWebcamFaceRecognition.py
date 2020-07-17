from imutils.video import VideoStream
import cv2
import joblib
import imutils
import config
from faceDetector import FaceDetector
from imageRender import ImageRender
from faceRecognitionModel import FaceRecognitionModel

known_face_encodings = joblib.load("x_train.dat")
known_face_labels = joblib.load("y_train.dat")

webcam = VideoStream(src=config.WEBCAM_SOURCE).start()
detector = FaceDetector("vj")
imageRender = ImageRender()
faceRecognitionModel = FaceRecognitionModel(known_face_labels, known_face_encodings)

while True:
    frame = webcam.read()
    frame = imutils.resize(frame, width=config.WEBCAM_FRAME_RESIZE)
    face_locations = detector.detectBBoxes(image=frame, scale_factor=config.VJ_SCALE_FACTOR,
                                           min_neighbors=config.VJ_MIN_NEIGHBORS,
                                           face_min_size=config.VJ_FACE_MIN_SIZE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for face_location in face_locations:
        predicted_label = faceRecognitionModel.predictFaceLabel(image=rgb, face_location=face_location,
                                                                tolerance=config.FACE_COMPARISON_TOLERANCE)
        imageRender.drawBBox(frame, face_location, config.CV2_BBOX_FILL, config.BBOX_WIDTH)
        imageRender.putLabel(frame, predicted_label, face_location, config.CV2_TEXT_FILL,
                             config.CV2_FONT_SCALE)

    cv2.imshow("Online Webcam", frame)
    cv2.waitKey(1)
