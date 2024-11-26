import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os

# Define known faces with profiles
known_faces = [
    {"path": "./For_training/Ronaldo.png", "name": "Cristiano Ronaldo",
     "profile": {"Team": "Al-Nassr", "Goals": 850, "Country": "Portugal"}},
    {"path": "./For_training/MESSI.jpg", "name": "Lionel Messi",
     "profile": {"Team": "Inter Miami", "Goals": 800, "Country": "Argentina"}},
    {"path": "./For_training/De Bruyne.png", "name": "Kevin De Bruyne",
     "profile": {"Team": "Manchester City", "Goals": 150, "Country": "Belgium"}},
    {"path": "./For_training/Halland.png", "name": "Erling Halland",
     "profile": {"Team": "Manchester City", "Goals": 200, "Country": "Norway"}},
    {"path": "./For_training/lewandovski.png", "name": "Robert Lewandovski",
     "profile": {"Team": "Barcelona", "Goals": 500, "Country": "Poland"}},
    {"path": "./For_training/Mahrez.png", "name": "Riyad Mahrez",
     "profile": {"Team": "Al-Ahli", "Goals": 150, "Country": "Algeria"}},
    {"path": "./For_training/Mané.jpg", "name": "Sadio Mané",
     "profile": {"Team": "Al-Nassr", "Goals": 200, "Country": "Senegal"}},
    {"path": "./For_training/Mbappé.png", "name": "Kylian Mbappé",
     "profile": {"Team": "PSG", "Goals": 250, "Country": "France"}},
    {"path": "./For_training/Neymar.png", "name": "Neymar JR",
     "profile": {"Team": "Al-Hilal", "Goals": 400, "Country": "Brazil"}},
    {"path": "./For_training/Van dyke.png", "name": "Virgil Van Dijk",
     "profile": {"Team": "Liverpool", "Goals": 50, "Country": "Netherlands"}}
]

# Load face encodings and names
known_face_encodings = []
known_face_names = []
player_profiles = {}


for face in known_faces:
    try:
        image = face_recognition.load_image_file(face["path"])
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(face["name"])
        player_profiles[face["name"]] = face["profile"]
    except Exception as e:
        print(f"Error processing {face['path']}: {e}")

# Assign unique colors to players
player_colors = {
    name: tuple(np.random.choice(range(256), size=3)) for name in known_face_names
}


def recognize_faces_in_image(image_path):
    """
    Recognize faces in a given image and draw bounding boxes with names.
    """
    try:
        unknown_image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        # Convert to PIL Image for drawing
        pil_image = Image.fromarray(unknown_image)
        draw = ImageDraw.Draw(pil_image)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Default to "Unknown" if no matches
            name = "Unknown"
            color = (255, 0, 0)  # Red for unknown
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = player_colors[name]
                # Print player stats to the terminal
                print(f"Recognized: {name}")
                profile = player_profiles.get(name, {})
                for key, value in profile.items():
                    print(f"{key}: {value}")
                print("-" * 30)

            # Confidence Score
            confidence = 100 - face_distances[best_match_index] * 100 if name != "Unknown" else 0
            label = f"{name} ({confidence:.2f}%)" if name != "Unknown" else name

            # Draw bounding box
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)

            # Draw label
            text_size = draw.textbbox((0, 0), label)[2:]
            text_height = text_size[1]
            draw.rectangle([(left, bottom - text_height - 10), (right, bottom)], fill=color)
            draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255, 255))

        # Save or display the result
        pil_image.show()
        pil_image.save("output.jpg")
        print("Recognition complete. Output saved as 'output.jpg'.")
    except Exception as e:
        print(f"Error during recognition: {e}")


if __name__ == "__main__":
    # Input handling
    image_number = input("Please enter the image number: ")
    image_path = f'./For_test/{image_number}.jpg'
    if not os.path.exists(image_path):
        print("Image not found. Please check the input.")
    else:
        recognize_faces_in_image(image_path)
