from io import text_encoding

import face_recognition
from PIL import Image, ImageDraw, ImageFont
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

Mahrez_image = face_recognition.load_image_file("./For_training/Mahrez.png")
Mahrez_face_encoding = face_recognition.face_encodings(Mahrez_image)[0]

Mane_image = face_recognition.load_image_file("./For_training/Mané.jpg")
Mane_face_encoding = face_recognition.face_encodings(Mane_image)[0]

Mbappe_image = face_recognition.load_image_file("./For_training/Mbappé.png")
Mbappe_face_encoding = face_recognition.face_encodings(Mbappe_image)[0]

Neymar_image = face_recognition.load_image_file("./For_training/Neymar.png")
Neymar_face_encoding = face_recognition.face_encodings(Neymar_image)[0]

Van_dyke_image = face_recognition.load_image_file("./For_training/Van dyke.png")
Van_dyke_face_encoding = face_recognition.face_encodings(Van_dyke_image)[0]


known_face_encodings = [ronaldo_face_encoding,
                        messi_face_encoding,
                        de_bruyne_face_encoding,
                        halland_face_encoding,
                        lewandovski_face_encoding,
                        Mahrez_face_encoding,
                        Mane_face_encoding,
                        Mbappe_face_encoding,
                        Neymar_face_encoding,
                        Van_dyke_face_encoding]
# player name
known_face_names = ["Cristiano Ronaldo",
                    "Lionel Messi",
                    "Kevin De Bruyne",
                    "Erling Halland",
                    "Robert Lewandovski",
                    "Ryath Mahrez",
                    "Sadio Mané",
                    "kylin Mbappé"
                    "Neymar JR",
                    "Virgil Van Dijk"]
#
    #print(best_match_index)
    #print(matches)

image = input("please enter the image number : ")
if image == "4":
    print("this image is not available")
    exit()
unknown_image = face_recognition.load_image_file(f'./For_test/{image}.jpg')

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image,face_locations)

pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)
for(top, right ,bottom , left ),face_encoding in zip (face_locations,face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
    name = "Unknown"
    face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)

    #print(face_distance)
    best_match_index = np.argmin(face_distance)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]


    draw.rectangle(((left , top) , (right , bottom)), outline = (0,0,255))
    text_width, text_height = draw.textbbox((0, 0), name)[2:]
    draw.rectangle(((left , bottom - text_height - 10),(right ,bottom)) , fill = (0,0,255) , outline = (0,0,255))
    draw.text((left + 6 , bottom - text_height - 5),name,fill = (255,255,255,255))

## help me
#pil_image.save("output.jpg")

del draw
pil_image.show()