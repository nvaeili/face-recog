import face_recognition as fr
import matplotlib.pyplot as plt
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

Tk().withdraw()
load_image = askopenfilename()

target_image = fr.load_image_file(load_image)
target_encoding = fr.face_encodings(target_image)[0]  # Get the first face encoding

def encode_faces(folder):
    list_people_encoding = []

    for filename in os.listdir(folder):
        known_image = fr.load_image_file(os.path.join(folder, filename))
        known_encoding = fr.face_encodings(known_image)[0]
        
        list_people_encoding.append((filename, known_encoding))

    return list_people_encoding

def find_target_face():
    face_location = fr.face_locations(target_image)

    if not face_location:
        print("No faces found in the target image.")
        return

    known_people = encode_faces('people/')

    for location in face_location:
        matched_person = None
        for person in known_people:
            filename = person[0]
            encoded_face = person[1]
            
            if fr.compare_faces([encoded_face], target_encoding, tolerance=0.55)[0]:
                matched_person = filename
                break
        
        if matched_person:
            create_frame(location, matched_person)

def create_frame(location, label):
    top, right, bottom, left = location

    cv.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(target_image, (left, bottom), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(target_image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

def render_image():
    rgb_img = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title('Face Recognition')
    plt.axis('off')
    plt.show()

find_target_face()
render_image()
