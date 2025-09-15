import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# Load image
img1 = cv2.imread('rabin3.jpeg')
#img1 = cv2.resize(img1, (640, 480))
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

# Analyze with DeepFace
predictions = DeepFace.analyze(img_path=img1, actions=["age", "gender", "emotion"], enforce_detection=False)

# Face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw rectangle + put text
font = cv2.FONT_HERSHEY_SIMPLEX

# DeepFace sometimes returns a dict, sometimes a list of dicts
if isinstance(predictions, list):
    results = predictions
else:
    results = [predictions]

for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3)

    if i < len(results):
        dominant_emotion = results[i].get('dominant_emotion', 'N/A')
        gender = results[i].get('dominant_gender', 'N/A')
        age = results[i].get('age', 'N/A')

        text = f"{dominant_emotion} | {gender} | {age}"
        #text = f"{dominant_emotion} | {gender} | {age}"
        cv2.putText(img1, f"Emotion: {dominant_emotion}", (x, y - 40), font, 0.6, (0, 0, 255), 2)
        cv2.putText(img1, f"Gender: {gender}", (x, y - 20), font, 0.6, (0, 255, 0), 2)
        cv2.putText(img1, f"Age: {age}", (x, y), font, 0.6, (255, 0, 0), 2)

# Show with matplotlib
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Also show with OpenCV
cv2.imshow("Detected Faces", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
