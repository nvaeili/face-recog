import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np

def get_encoded_faces():
    encoded = {}

    for dirparth, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f )
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

def classify_face(im):
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
            cv2.rectangle(img, (left-20, bottom-15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left-20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    new_width = 800  # Set the desired width for resized image
    new_height = int(img.shape[0] * (new_width / img.shape[1]))  # Calculate height while maintaining aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height))

    cv2.imshow('Facial Recognition', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return face_names

print(classify_face("test.png"))