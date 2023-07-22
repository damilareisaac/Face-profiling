from deepface.basemodels import VGGFace
import numpy as np
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Activation
from config import RACE_MODEL_WEIGHT_PATH
from models.utils import transform_face_array


class RaceModel:
    def __init__(self):
        self.model = self.load_model()
        self.race_labels = [
            "asian",
            "indian",
            "black",
            "white",
            "middle eastern",
            "latino hispanic",
        ]

    def predict_race(self, face_image):
        image_preprocesing = transform_face_array(face_image)
        race_predictions = self.model.predict(image_preprocesing)[0, :]
        result_race = self.race_labels[np.argmax(race_predictions)]
        return result_race

    def load_model(self):
        model = VGGFace.baseModel()

        classes = 6
        base_model_output = Convolution2D(classes, (1, 1), name="predictions")(
            model.layers[-4].output
        )
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)

        race_model = Model(inputs=model.input, outputs=base_model_output)

        race_model.load_weights(RACE_MODEL_WEIGHT_PATH)
        return race_model
