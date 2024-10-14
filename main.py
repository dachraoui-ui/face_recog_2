import face_recognition
from PIL import Image, ImageDraw
import numpy as np


ronaldo_image = face_recognition.load_image_file("./For_training/Ronaldo.png")
ronaldo_face_encoding = face_recognition.face_encodings(ronaldo_image)[0]

print(ronaldo_face_encoding)

messi_image = face_recognition.load_image_file("./For_training/MESSI.jpg")
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]

print(messi_face_encoding)