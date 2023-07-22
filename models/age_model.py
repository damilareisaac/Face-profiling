from deepface.basemodels import VGGFace
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Activation
import numpy as np
from .utils import transform_face_array

from config import AGE_MODEL_WEIGHT_PATH


class AgeModel:
    def __init__(self):
        self.model = self.load_model()
        self.output_indexes = np.array([i for i in range(0, 101)])

    def load_model(self):
        model = VGGFace.baseModel()
        # --------------------------
        classes = 101
        base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
            model.layers[-4].output
        )
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        # --------------------------
        age_model = Model(inputs=model.input, outputs=base_model_output)
        # --------------------------
        # load weights
        age_model.load_weights(AGE_MODEL_WEIGHT_PATH)
        return age_model

    def predict_age(self, face_image):
        image_preprocesing = transform_face_array(face_image)
        age_predictions = self.model.predict(image_preprocesing)[0, :]
        result_age = self.find_apparent_age(age_predictions)
        return result_age

    def find_apparent_age(self, age_predictions):
        apparent_age = np.sum(age_predictions * self.output_indexes)
        return apparent_age
