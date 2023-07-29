import face_recognition
import traceback
import numpy as np
import cv2
import os
import config


class rec:
    def __init__(self):
        self.db_names, self.db_features = self.load_images_to_database()

    def detect_face(self, img):
        temp = face_recognition.face_locations(img)
        print("Face Recognition: ", temp)
        return temp

    def compare_faces(
        self,
        face_encodings,
        db_features,
        db_names,
    ):
        match_name = []
        names_temp = db_names
        feats_temp = db_features

        for face_encoding in face_encodings:
            try:
                dist = face_recognition.face_distance(
                    feats_temp,
                    face_encoding,
                )
            except Exception as e:
                dist = face_recognition.face_distance(
                    [feats_temp],
                    face_encoding,
                )
            index = np.argmin(dist)
            if dist[index] <= 0.6:
                match_name = match_name + [names_temp[index]]
            else:
                match_name = match_name + ["unknow"]
        return match_name

    def get_features(self, img, box):
        return face_recognition.face_encodings(img, box)

    def recognize_face(self, im):
        try:
            # detectar rostro
            box_faces = self.detect_face(im)
            # condiconal para el caso de que no se detecte rostro
            if not box_faces:
                res = {
                    "status": "ok",
                    "faces": [],
                    "names": [],
                }
                return res
            else:
                if not self.db_names:
                    res = {
                        "status": "ok",
                        "faces": box_faces,
                        "names": ["unknow"] * len(box_faces),
                    }
                    return res
                else:
                    # (continua) extraer features
                    actual_features = self.get_features(im, box_faces)
                    # comparar actual_features con las que estan almacenadas en la base de datos
                    match_names = self.compare_faces(
                        actual_features,
                        self.db_features,
                        self.db_names,
                    )
                    # guardar
                    res = {
                        "status": "ok",
                        "faces": box_faces,
                        "names": match_names,
                    }
                    return res
        except Exception as e:
            error = "".join(
                traceback.format_exception(
                    type(e),
                    e,
                    tb=e.__traceback__,
                )
            )
            res = {
                "status": "error: " + str(error),
                "faces": [],
                "names": [],
            }
            return res

    def recognize_face2(self, im, box_faces):
        try:
            if not self.db_names:
                res = "unknow"
                return res
            else:
                # (continua) extraer features
                actual_features = self.get_features(im, box_faces)
                # comparar actual_features con las que estan almacenadas en la base de datos
                match_names = self.compare_faces(
                    actual_features,
                    self.db_features,
                    self.db_names,
                )
                # guardar
                res = match_names
                return res
        except Exception as e:
            print(e)
            res = []
            return res

    def load_images_to_database(self):
        list_images = os.listdir(str(config.IMAGE_DIR))
        list_images = [
            File for File in list_images if File.endswith((".jpg", ".jpeg", "JPEG"))
        ]

        name = []
        feats = []

        for file_name in list_images:
            im = cv2.imread(
                str(config.IMAGE_DIR / file_name),
            )

            box_face = self.detect_face(im)
            feat = self.get_features(im, box_face)
            if len(feat) != 1:
                continue
            else:
                new_name = file_name.split(".")[0]
                if new_name == "":
                    continue
                name.append(new_name)
                if len(feats) == 0:
                    feats = np.frombuffer(
                        feat[0],
                        dtype=np.float64,
                    )
                else:
                    feats = np.vstack(
                        (
                            feats,
                            np.frombuffer(
                                feat[0],
                                dtype=np.float64,
                            ),
                        )
                    )
        return name, feats


def bounding_box(
    img,
    box,
    match_name=[],
):
    for i in np.arange(len(box)):
        x0, y0, x1, y1 = box[i]
        img = cv2.rectangle(
            img,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            3,
        )
        if not match_name:
            continue
        else:
            cv2.putText(
                img,
                match_name[i],
                (x0, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
    return img
