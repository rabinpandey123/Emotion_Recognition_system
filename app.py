from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
from tensorflow.keras.models import load_model
import base64
import io
from PIL import Image
import threading
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load FER model
try:
    fer_model = load_model("models/fer_model.h5")
    emotion_labels_fer = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    logger.info("FER model loaded successfully")
except Exception as e:
    logger.error(f"Error loading FER model: {e}")
    fer_model = None

# Global variables for video processing
video_processing = False
video_results = {"status": "inactive", "message": "Video processing not started"}
video_capture = None


def analyze_with_deepface(img_array):
    try:
        # Try multiple backends for better accuracy
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn']
        results = []

        for backend in backends:
            try:
                result = DeepFace.analyze(
                    img_path=img_array,
                    actions=["emotion", "age", "gender"],
                    enforce_detection=False,
                    detector_backend=backend,
                    align=True
                )
                if result and len(result) > 0:
                    results.extend(result if isinstance(result, list) else [result])
                    logger.info(f"Successfully used {backend} backend")
                    break  # Use the first successful backend
            except Exception as e:
                logger.warning(f"DeepFace with {backend} backend failed: {e}")
                continue

        if not results:
            # Fallback to default with no detection enforcement
            results = DeepFace.analyze(
                img_path=img_array,
                actions=["emotion", "age", "gender"],
                enforce_detection=False,
                detector_backend='opencv'
            )

        return results if isinstance(results, list) else [results]

    except Exception as e:
        logger.error(f"All DeepFace backends failed: {e}")
        return []


def analyze_with_fer(frame):
    fer_results = []
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)

        # Multiple face detection methods
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no faces found, try with more sensitive parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        for (x, y, w, h) in faces:
            # Extract face region with padding
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)

            roi_gray = gray[y1:y2, x1:x2]

            # Resize to match model input
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Apply additional contrast enhancement
            roi_gray = cv2.equalizeHist(roi_gray)

            # Normalize with the same method used during training
            roi_gray = roi_gray.astype("float32") / 255.0

            # Expand dimensions
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            # Predict with temperature scaling for better confidence
            preds = fer_model.predict(roi_gray, verbose=0)[0]

            # Apply temperature scaling to calibrate confidence
            temperature = 0.5  # Lower temperature for more peaked distribution
            scaled_preds = np.exp(preds / temperature) / np.sum(np.exp(preds / temperature))

            # Get the emotion with highest probability
            emotion_idx = np.argmax(scaled_preds)
            label = emotion_labels_fer[emotion_idx]
            confidence = float(scaled_preds[emotion_idx]) * 100

            # Apply confidence threshold - don't show low confidence results
            if confidence > 40:  # Only show results with >40% confidence
                fer_results.append({
                    "emotion": label,
                    "confidence": round(confidence, 2),
                    "bbox": [int(x), int(y), int(w), int(h)]
                })
            else:
                # For low confidence, show "uncertain" instead of likely wrong prediction
                fer_results.append({
                    "emotion": "uncertain",
                    "confidence": round(confidence, 2),
                    "bbox": [int(x), int(y), int(w), int(h)]
                })

    except Exception as e:
        logger.error(f"FER analysis error: {e}")

    return fer_results


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        # Get image from request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"success": False, "error": "No image provided"})

        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Analyze with improved DeepFace
        deepface_results = analyze_with_deepface(img_array)

        # Analyze with improved FER model
        fer_results = []
        if fer_model is not None:
            fer_results = analyze_with_fer(img_array)

        # Format DeepFace results
        formatted_df_results = []
        for res in deepface_results:
            bbox = [0, 0, 0, 0]
            if 'region' in res:
                region = res['region']
                bbox = [region.get('x', 0), region.get('y', 0),
                        region.get('w', 0), region.get('h', 0)]

            formatted_df_results.append({
                "emotion": res.get('dominant_emotion', 'N/A'),
                "age": res.get('age', 'N/A'),
                "gender": res.get('dominant_gender', 'N/A'),
                "bbox": bbox
            })

        return jsonify({
            "success": True,
            "deepface": formatted_df_results,
            "fer": fer_results
        })

    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route('/api/diagnose_fer', methods=['POST'])
def diagnose_fer_model():
    try:
        # Get image from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Process for FER model
        gray = cv2.resize(gray, (48, 48))
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=-1)
        gray = np.expand_dims(gray, axis=0)

        # Get predictions
        preds = fer_model.predict(gray, verbose=0)[0]

        # Return detailed predictions
        detailed_predictions = {}
        for i, emotion in enumerate(emotion_labels_fer):
            detailed_predictions[emotion] = float(preds[i] * 100)

        return jsonify({
            "success": True,
            "predictions": detailed_predictions,
            "dominant_emotion": emotion_labels_fer[np.argmax(preds)],
            "confidence": float(np.max(preds)) * 100
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route('/api/start_video', methods=['POST'])
def start_video_analysis():
    global video_processing, video_results, video_capture

    if video_processing:
        return jsonify({"success": False, "message": "Video processing already running"})

    video_processing = True
    video_results = {"status": "starting", "message": "Initializing video processing"}

    # Start a thread to process video frames
    try:
        thread = threading.Thread(target=process_video_frames, daemon=True)
        thread.start()
        return jsonify({"success": True, "message": "Video processing started"})
    except Exception as e:
        video_processing = False
        logger.error(f"Error starting video thread: {e}")
        return jsonify({"success": False, "message": f"Error starting video: {e}"})


@app.route('/api/stop_video', methods=['POST'])
def stop_video_analysis():
    global video_processing, video_capture

    video_processing = False

    # Release the camera if it's open
    if video_capture and video_capture.isOpened():
        video_capture.release()
        video_capture = None

    return jsonify({"success": True, "message": "Video processing stopped"})


@app.route('/api/video_results', methods=['GET'])
def get_video_results():
    return jsonify({
        "success": True,
        "results": video_results
    })


def process_video_frames():
    global video_processing, video_results, video_capture

    # Try different camera indices (0 is usually the default)
    camera_indices = [0, 1, 2]
    cap = None

    for camera_index in camera_indices:
        try:
            # Try with DSHOW backend first (works better on Windows)
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                logger.info(f"Successfully opened camera index {camera_index} with DSHOW backend")
                break
            else:
                cap.release()
                # Try with default backend
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    logger.info(f"Successfully opened camera index {camera_index} with default backend")
                    break
                else:
                    cap.release()
        except Exception as e:
            logger.error(f"Error opening camera index {camera_index}: {e}")
            if cap:
                cap.release()

    if cap is None or not cap.isOpened():
        video_results = {"status": "error", "message": "Cannot open any camera"}
        video_processing = False
        return

    video_capture = cap

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Allow camera to warm up
    time.sleep(2)

    frame_count = 0
    process_every_n_frames = 5  # Process every 5th frame

    video_results = {"status": "active", "message": "Camera active, processing frames"}

    while video_processing:
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Process only every nth frame
            if frame_count % process_every_n_frames != 0:
                continue

            # Analyze with DeepFace
            deepface_results = analyze_with_deepface(frame)

            # Analyze with FER model if available
            fer_results = []
            if fer_model is not None:
                fer_results = analyze_with_fer(frame)

            # Format DeepFace results
            if isinstance(deepface_results, list):
                df_results = deepface_results
            else:
                df_results = [deepface_results]

            formatted_df_results = []
            for i, res in enumerate(df_results):
                bbox = [0, 0, 0, 0]
                if 'region' in res:
                    region = res['region']
                    bbox = [region.get('x', 0), region.get('y', 0),
                            region.get('w', 0), region.get('h', 0)]

                formatted_df_results.append({
                    "emotion": res.get('dominant_emotion', 'N/A'),
                    "age": res.get('age', 'N/A'),
                    "gender": res.get('dominant_gender', 'N/A'),
                    "bbox": bbox
                })

            # Update results
            video_results = {
                "status": "active",
                "message": "Processing successful",
                "deepface": formatted_df_results,
                "fer": fer_results,
                "timestamp": time.time()
            }

            # Small delay
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in video processing loop: {e}")
            time.sleep(1)

    # Release camera when done
    try:
        cap.release()
        video_capture = None
    except:
        pass

    video_processing = False
    video_results = {"status": "inactive", "message": "Video processing stopped"}


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "fer_model_loaded": fer_model is not None
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)