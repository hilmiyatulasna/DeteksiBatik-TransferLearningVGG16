from flask import Flask, request, render_template, Response
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time

app = Flask(__name__)

# Path ke model
model_path = 'batik_classification/models/my_model.keras'
model = load_model(model_path)

# Fungsi untuk melakukan prediksi
def predict_batik(image):
    img_array = cv2.resize(image, (150, 150))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return predictions

# Classes for batik classification
classes = ['Batik Bali', 'Batik Betawi', 'Batik Celup', 'Batik Cendrawasih', 'Batik Ceplok', 'Batik Ciamis', 'Batik Garutan', 'Batik Gentongan', 'Batik Kawung', 'Batik Keraton', 'Batik Lasem', 'Batik Megamendung', 'Batik Parang', 'Batik Pekalongan', 'Batik Priangan', 'Batik Sekar', 'Batik Sidoluhur', 'Batik Sidomukti', 'Batik Sogan', 'Batik Tambal']  # Adjust as per your classes

def generate_frames():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("No Camera")

    while True:
        ret, image = cam.read()
        if not ret:
            break

        predictions = predict_batik(image)
        predicted_class = np.argmax(predictions[0])

        label = classes[predicted_class]
        cv2.putText(image, f'{label} - {predictions[0][predicted_class]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Menyimpan file yang diupload
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Membaca gambar dari file
            image = cv2.imread(file_path)
            if image is not None:
                # Melakukan prediksi
                predictions = predict_batik(image)
                # Ambil kelas prediksi (indeks kelas teratas)
                predicted_class = np.argmax(predictions[0])

                # Render template dengan hasil prediksi
                return render_template('upload.html', prediction=classes[predicted_class])
            else:
                return "Gagal memuat gambar", 400

    return render_template('upload.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
