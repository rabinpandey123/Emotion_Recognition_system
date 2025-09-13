import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("fer_model.h5")

# Emotion labels (must match your dataset folders order)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load image (change path to your image)
image_path = "angry_man.jpg"
frame = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Use OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48,48))
    roi_gray = roi_gray.astype("float") / 255.0
    roi_gray = np.expand_dims(roi_gray, axis=-1)   # channel dimension
    roi_gray = np.expand_dims(roi_gray, axis=0)    # batch dimension

    # Prediction
    preds = model.predict(roi_gray, verbose=0)[0]
    label = emotion_labels[np.argmax(preds)]
    confidence = np.max(preds)

    # Draw rectangle and text
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# Show result
cv2.imshow("Emotion Detection - Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
