from imutils.video import VideoStream
import cv2
import joblib
import config
from imageRender import ImageRender
from faceDetector import FaceDetector
from faceRecognitionModel import FaceRecognitionModel

known_face_encodings = joblib.load("x_train.dat")
known_face_labels = joblib.load("y_train.dat")
faceRecognitionModel = FaceRecognitionModel(known_face_labels, known_face_encodings)

webcam = VideoStream(src=config.WEBCAM_SOURCE).start()
detector = FaceDetector(config.FACE_DETECTOR_MODEL)
videoFrameRender = ImageRender()


while True:
    frame = webcam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = detector.detectBBoxes(image=rgb,
                                           number_times_to_upsample=config.FACE_DETECTOR_NUMBER_TIMES_UPSAMPLE)

    for face_location in face_locations:
        predicted_label = faceRecognitionModel.predictFaceLabel(image=rgb, face_location=face_location,
                                                                tolerance=config.FACE_COMPARISON_TOLERANCE)

        videoFrameRender.drawBBox(frame, face_location, config.CV2_BBOX_FILL, config.BBOX_WIDTH)
        videoFrameRender.putLabel(frame, predicted_label, face_location, config.CV2_TEXT_FILL,
                                  config.CV2_FONT_SCALE)

    cv2.imshow("Online Webcam", frame)
    cv2.waitKey(1)
