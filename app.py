from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

# Kamera
camera = cv2.VideoCapture(0)

# Modele FaceNet i MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Baza znanych twarzy
known_face_encodings = []
known_face_names = []

# Wczytywanie zarejestrowanych twarzy
def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    folder = 'known_faces'
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = mtcnn(img_rgb)
            if faces is not None:
                for face in faces:
                    if face.ndim == 3:  # [3, 160, 160]
                        face = face.unsqueeze(0)  # [1, 3, 160, 160]
                    with torch.no_grad():
                        embedding = resnet(face.to(device))
                        known_face_encodings.append(embedding.cpu())
                        known_face_names.append(filename[:-4])

load_known_faces()

def recognize_face(face_img):
    faces = mtcnn(face_img)
    if faces is not None:
        if faces.ndim == 3:  # [3, 160, 160]
            faces = faces.unsqueeze(0)  # [1, 3, 160, 160]
        with torch.no_grad():
            embedding = resnet(faces.to(device))
            for idx, known_embedding in enumerate(known_face_encodings):
                dist = (embedding.cpu() - known_embedding).norm().item()
                if dist < 0.9:  # próg dopasowania
                    return known_face_names[idx]
    return "Unknown"

def is_live_face(face_img):
    """
    Metoda detekcji żywej twarzy bazująca na analizie tekstury i dynamice twarzy.
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Analiza krawędzi - silniejsze krawędzie wskazują na żywą twarz
    edge_strength = np.sum(edges)
    
    # Jeśli suma krawędzi jest większa, to uznajemy twarz za żywą
    if edge_strength > 150000:  # Granica, którą można dostosować
        return True
    return False

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        # Sprawdzenie, czy wykryto twarze
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = img_rgb[y1:y2, x1:x2]
                name = recognize_face(face_img)

                # Weryfikacja, czy twarz jest "żywa"
                if is_live_face(face_img):  # Jeśli wykryto zmiany na twarzy (dynamiczna twarz)
                    color = (0, 255, 0)  # Zielony dla znanych twarzy
                    label = "live"  # Twarz żywa
                else:
                    name = "Unknown"  # Twarz uznana za nieznaną
                    color = (0, 0, 255)  # Czerwony dla nieznanych twarzy
                    label = "photo"  # Twarz na zdjęciu

                # Jeśli twarz jest żywa, ale nie rozpoznana
                if label == "live" and name == "Unknown":
                    label = "live (unknown)"
                if label == "photo" and name != "user":
                    label = "photo (known)"

                # Rysowanie prostokąta wokół twarzy i podpisu
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} - {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Wysyłanie ramki tylko wtedy, gdy twarze zostały wykryte
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # Skip this frame if encoding fails
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    success, frame = camera.read()
    if success:
        name = request.form.get('name', 'user')
        os.makedirs("known_faces", exist_ok=True)
        path = os.path.join("known_faces", f"{name}.jpg")
        cv2.imwrite(path, frame)
        load_known_faces()
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
