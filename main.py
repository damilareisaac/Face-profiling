from typing import Any, List
import face_recognition
import cv2
import imutils
import time
import numpy as np

from config import FONT_SIZE, STEP, TEXT_THICKNESS

cv2.namedWindow("Image Profile")
camera = cv2.VideoCapture(0)


def get_profile(frame):
    profiles = []
    face_features = {
        "bbx_frontal_face": [],
    }
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        for face_location in face_locations:
            x0, y1, x1, y0 = face_location
            bbx_frontal_face = np.array([y0, x0, y1, x1])

            face_features["bbx_frontal_face"] = bbx_frontal_face  # type: ignore
    profiles.append(face_features)
    return profiles


def add_rectangular_profile_box_to_face(profiles: List[Any], frame):
    for profile in profiles:
        bbx_frontal_face = profile.get("bbx_frontal_face")
        if len(bbx_frontal_face) <= 0:
            continue
        x0, y0, x1, y1 = bbx_frontal_face
        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        index = 0
        for key, val in profile.items():
            try:
                cv2.putText(
                    frame,
                    f"{key}: {val}",
                    (x0, y0 - STEP - 10 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE,
                    (0, 255, 0),
                    TEXT_THICKNESS,
                )
                index += 1
            except Exception as e:
                print(e)

        return frame


while True:
    start_time = time.time()
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=720)

    end_time = time.time() - start_time
    FPS = 1 / end_time

    profiles = get_profile(frame)
    annotated_images = add_rectangular_profile_box_to_face(profiles, frame)

    cv2.putText(
        frame,
        f"FPS: {round(FPS,3)}",
        (10, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.imshow("Face info", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
