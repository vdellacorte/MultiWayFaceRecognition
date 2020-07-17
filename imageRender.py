import cv2


class ImageRender:
    # def __init__(self):

    def drawBBox(self, image, position, fill, width):
        top, right, bottom, left = position
        cv2.rectangle(image, (left, top), (right, bottom), fill, width)
        return image

    def putLabel(self, image, text, position, text_fill, font_scale):
        top, right, bottom, left = position
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_fill)
        return image
