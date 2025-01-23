import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained CNN model
model = tf.keras.models.load_model('accident_detection_model.h5')

# Open video feed from the drone's camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame
    img = cv2.resize(frame, (64, 64))  # Resize to the model's expected input size
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension for prediction
    
    # Get model prediction (0 = No Accident, 1 = Accident)
    prediction = model.predict(img)
    
    if prediction[0] > 0.5:
        print("Accident detected!")
        send_alert()

    # Display the live feed for debugging
    cv2.imshow("Drone Camera Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
=
cap.release()
cv2.destroyAllWindows()
