import json
import sys

import cv2
import face_recognition
import numpy as np

# Following script has been borrowed from https://github.com/ageitgey/face_recognition

input_movie = cv2.VideoCapture(sys.argv[1])
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 1, (1280, 720))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    if frame_number % 24 != 0:
        continue

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    print(json.dumps([frame_number, face_locations, face_encodings], cls=NumpyEncoder))


# All done!
input_movie.release()
cv2.destroyAllWindows()
