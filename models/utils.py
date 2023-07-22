import cv2
import numpy as np
from keras.preprocessing import image


def transform_face_array(
    face_array,
    grayscale=False,
    target_size=(224, 224),
):
    detected_face = face_array
    if grayscale == True:
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
    detected_face = cv2.resize(detected_face, target_size)
    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    # normalize input in [0, 1]
    img_pixels /= 255
    return img_pixels
