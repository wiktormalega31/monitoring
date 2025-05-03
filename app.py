from flask import Flask, render_template, Response, request
import cv2
import face_recognition
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

# Wczytaj wcześniej zarejestrowaną twarz (jeśli istnieje)
def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    if os.path.exists("known_faces/trusted.jpg"):
        image = face_recognition.load_image_file("known_faces/trusted.jpg")
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append("Trusted")

load_known_faces()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Zmniejsz obraz do detekcji
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Wykryj twarze i zakoduj je
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    name = known_face_names[matches.index(True)]

            # Skaluje z powrotem do oryginalnego rozmiaru
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register_face():
    ret, frame = camera.read()
    if ret:
        os.makedirs("known_faces", exist_ok=True)
        cv2.imwrite("known_faces/trusted.jpg", frame)
        load_known_faces()
    return ('', 204)  # brak odpowiedzi, tylko akcja

if __name__ == '__main__':
    app.run(debug=True)
