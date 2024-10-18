import face_recognition
from PIL import Image, ImageDraw
import numpy as np


ronaldo_image = face_recognition.load_image_file("./For_training/Ronaldo.png")
ronaldo_face_encoding = face_recognition.face_encodings(ronaldo_image)[0]

messi_image = face_recognition.load_image_file("./For_training/MESSI.jpg")
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]

de_bruyne_image = face_recognition.load_image_file("./For_training/De Bruyne.png")
de_bruyne_face_encoding = face_recognition.face_encodings(de_bruyne_image)[0]

halland_image = face_recognition.load_image_file("./For_training/Halland.png")
halland_face_encoding = face_recognition.face_encodings(halland_image)[0]

lewandovski_image = face_recognition.load_image_file("./For_training/lewandovski.png")
lewandovski_face_encoding = face_recognition.face_encodings(lewandovski_image)[0]

known_face_encodings = [ronaldo_face_encoding,
                        messi_face_encoding,
                        de_bruyne_face_encoding,
                        halland_face_encoding,
                        lewandovski_face_encoding]

known_face_names = ["Cristiano Ronaldo",
                    "Lionel Messi",
                    "Kevin De Bruyne",
                    "Erling Halland",
                    "Robert Lewandovski"]


