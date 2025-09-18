from tensorflow.keras.models import load_model

model = load_model("models/fer_model.h5")
print("Model loaded successfully!")
