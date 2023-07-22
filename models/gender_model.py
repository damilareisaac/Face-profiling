from config import GENDER_MODEL_WEIGHT_PATH
from models.utils import transform_face_array
from deepface.basemodels import VGGFace
import numpy as np
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Activation


class GenderModel:
    def __init__(self):
        self.model = self.load_model()

    def predict_gender(self, face_image):
        image_preprocesing = transform_face_array(face_image)
        gender_predictions = self.model.predict(image_preprocesing)[0, :]
        result_gender = "Female" if np.argmax(gender_predictions) == 0 else "Male"
        return str(result_gender)

    def load_model(self):
        model = VGGFace.baseModel()
        classes = 2
        base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
            model.layers[-4].output
        )
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        gender_model = Model(inputs=model.input, outputs=base_model_output)
        gender_model.load_weights(GENDER_MODEL_WEIGHT_PATH)
        return gender_model
