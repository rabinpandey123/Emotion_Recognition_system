import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load pre-trained FER model (you can use your own trained model)
# For demonstration, let's assume you have 'fer_model.h5'
model = load_model('fer_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MTCNN face detector
detector = MTCNN()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)  # avoid negative indices
        face_img = rgb_frame[y:y+height, x:x+width]

        try:
            # Preprocess face for model
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)

            # Predict emotion
            predictions = model.predict(face_img)
            emotion_idx = np.argmax(predictions)
            emotion_label = emotion_labels[emotion_idx]

            # Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        except:
            continue

    cv2.imshow('Real-Time FER', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
