import json
import sys

import face_recognition
import numpy as np


class KnownFaces:
    def __init__(self):
        self.known_faces_names = []
        self.known_faces_encodings = np.array([])
        self.known_faces_encodings.resize((0, 128))

    def learn_face(self, name, encoding):
        self.known_faces_names.append(name)
        self.known_faces_encodings = np.vstack((self.known_faces_encodings, encoding))

    def match(self, face_encoding):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
        name = 'Unknown'
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = self.known_faces_names[first_match_index]

        return name


def build_known_faces_dict():
    known_faces = KnownFaces()
    base_path = 'data/office/faces'
    faces_map = {
        'Pam': base_path + '/Pam.png',
        'Dwight': base_path + '/Dwight.png',
        'Jim': base_path + '/Jim.png',
        'Michael': base_path + '/Michael.png',
        'Roy': base_path + '/Roy.png',
        'Ryan': base_path + '/Ryan.png',
        'Oscar': base_path + '/Oscar.png',
        'Angela': base_path + '/Angela.png',
        'Phyllis': base_path + '/Phyllis.png',
        'Kevin': base_path + '/Kevin.png',
        'Jane': base_path + '/Jane.png',
        'Creed': base_path + '/Creed.png',
        'Stanley': base_path + '/Stanley.png',
        'Kelly': base_path + '/Kelly.png',
        'Meredith': base_path + '/Meredith.png',
        'Toby': base_path + '/Toby.png',
        'Darryl': base_path + '/Darryl.png',
        'Bob Vance': base_path + '/Bob Vance.png',
        'Katy': base_path + '/Katy.png',
    }
    for name, file_name in faces_map.items():
        image = face_recognition.load_image_file(file_name)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.learn_face(name, encoding)

    return known_faces


def main():
    known_faces = build_known_faces_dict()
    file_name = sys.argv[1]

    with open(file_name) as f:
        content = f.readlines()
        for line in content:
            [frame_number, face_locations, face_encodings] = json.loads(line)
            frame_faces = []
            for face_location, face_encoding in zip(face_locations, face_encodings):
                face_name = known_faces.match(face_encoding)
                frame_faces.append(face_name)
            print(json.dumps([frame_number, frame_faces]))


main()
