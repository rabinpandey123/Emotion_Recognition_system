import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

frame_count = 0
process_every_n_frames = 5  # Analyze every 5th frame for speed

# Store last results for each face
face_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detect faces in current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Update face results every N frames
    if frame_count % process_every_n_frames == 0:
        face_results = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(
                    face_img,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )[0]  # get first result
                face_results.append({
                    "pos": (x, y, w, h),
                    "age": result['age'],
                    "gender": result['dominant_gender'],
                    "emotion": result['dominant_emotion']
                })
            except:
                face_results.append({
                    "pos": (x, y, w, h),
                    "age": "N/A",
                    "gender": "N/A",
                    "emotion": "N/A"
                })

    # Draw rectangles and labels for each face
    for face in face_results:
        x, y, w, h = face['pos']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{face['emotion']} | {face['gender']} | {face['age']}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Show video
    cv2.imshow("Live Emotion | Gender | Age - Multiple Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
