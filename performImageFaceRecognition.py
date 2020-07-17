import joblib
import face_recognition
import config
from PIL import ImageDraw, Image, ImageFont
from faceRecognitionModel import FaceRecognitionModel
from faceDetector import FaceDetector

known_face_encodings = joblib.load("x_train.dat")
known_face_labels = joblib.load("y_train.dat")
faceRecognitionModel = FaceRecognitionModel(known_face_labels, known_face_encodings)

unknown_image = face_recognition.load_image_file(config.TESTING_IMAGE)
detector = FaceDetector(config.FACE_DETECTOR_MODEL)
face_locations = detector.detectBBoxes(unknown_image)
print("Found {} number of faces".format(len(face_locations)))

pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)
font = ImageFont.truetype("arial", config.BBOX_TEXT_SIZE)

for face_location in face_locations:
    predicted_label = faceRecognitionModel.predictFaceLabel(image=unknown_image, face_location=face_location,
                                                            tolerance=config.FACE_COMPARISON_TOLERANCE)

    top, right, bottom, left = face_location
    draw.rectangle((left, top, right, bottom), outline=config.BBOX_FILL, width=config.BBOX_WIDTH)
    y = top - 15 if top - 15 > 15 else top + 15
    draw.text((left, y), predicted_label, fill=config.BBOX_TEXT_FILL, font=font)

pil_image.show()
