from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

camera = cv2.VideoCapture(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

known_face_encodings = []
known_face_names = []

filter_settings = {
    "contrast": 1.3,
    "brightness": 30,
    "blur": 3
}

def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    folder = 'known_faces'
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img_rgb)
            if face is not None:
                if face.ndim == 3:
                    face = face.unsqueeze(0)
                with torch.no_grad():
                    embedding = resnet(face.to(device))
                    known_face_encodings.append(embedding.cpu())
                    known_face_names.append(filename[:-4])

load_known_faces()

def recognize_face(face_img):
    face = mtcnn(face_img)
    if face is not None:
        if face.ndim == 3:
            face = face.unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(face.to(device))
            for idx, known_embedding in enumerate(known_face_encodings):
                dist = (embedding.cpu() - known_embedding).norm().item()
                if dist < 0.5:  # próg podobieństwa
                    return known_face_names[idx]
    return "Unknown"

def is_live_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var > 50  # większe = ostrzejsze = żywa twarz

def apply_filters(frame):
    f = filter_settings
    frame = cv2.convertScaleAbs(frame, alpha=f["contrast"], beta=f["brightness"])
    if f["blur"] > 1:
        k = f["blur"] if f["blur"] % 2 == 1 else f["blur"] + 1
        frame = cv2.GaussianBlur(frame, (k, k), 0)
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = apply_filters(frame)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_img = img_rgb[y1:y2, x1:x2]
                name = recognize_face(face_img)
                live = is_live_face(face_img)

                if live and name != "Unknown":
                    label = f"{name} (live)"
                    color = (0, 255, 0)
                elif live:
                    label = "Live"
                    color = (255, 255, 0)
                else:
                    label = "Photo"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', filters=filter_settings)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name', 'user').strip()
    if name:
        os.makedirs("known_faces", exist_ok=True)
        i = 1
        while os.path.exists(f"known_faces/{name}_{i}.jpg"):
            i += 1
        path = f"known_faces/{name}_{i}.jpg"
        success, frame = camera.read()
        if success:
            cv2.imwrite(path, frame)
            load_known_faces()
    return redirect(url_for('index'))

@app.route('/clear_faces', methods=['POST'])
def clear_faces():
    for file in os.listdir('known_faces'):
        os.remove(os.path.join('known_faces', file))
    load_known_faces()
    return redirect(url_for('index'))

@app.route('/update_filters', methods=['POST'])
def update_filters():
    filter_settings["contrast"] = float(request.form.get("contrast", 1.3))
    filter_settings["brightness"] = int(request.form.get("brightness", 30))
    filter_settings["blur"] = int(request.form.get("blur", 3))
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
