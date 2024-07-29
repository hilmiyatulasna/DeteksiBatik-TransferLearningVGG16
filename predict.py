import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load VGG16 model
model_path = 'batik_classification/models/my_model.keras'
model = load_model(model_path)

# Function to predict batik
def predict_batik(image):
    img_array = cv2.resize(image, (150, 150))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    return predictions

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception("No Camera")

# Classes for batik classification
classes = ['Batik Bali', 'Batik Betawi', 'Batik Celup', 'Batik Cendrawasih', 'Batik Ceplok', 'Batik Ciamis', 'Batik Garutan', 'Batik Gentongan', 'Batik Kawung', 'Batik Keraton', 'Batik Lasem', 'Batik Megamendung', 'Batik Parang', 'Batik Pekalongan', 'Batik Priangan', 'Batik Sekar', 'Batik Sidoluhur', 'Batik Sidomukti', 'Batik Sogan', 'Batik Tambal']  # Adjust as per your classes

while True:
    ret, image = cam.read()
    if not ret:
        break

    start_time = time.time()
    predictions = predict_batik(image)
    predicted_class = np.argmax(predictions[0])
    end_time = time.time()
    
    # Display the prediction on the frame
    label = classes[predicted_class]
    cv2.putText(image, f'{label} - {predictions[0][predicted_class]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Batik Detection", image)
    
    print("Time taken: ", end_time - start_time)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()