import cv2
import time
import numpy as np
import tensorflow.lite as tflite
import requests
import smtplib
from email.message import EmailMessage

# ESP32-CAM Stream URL
ESP32_URL = "http://192.168.229.103/cam-hi.jpg"  # Replace <ESP32_IP> with your ESP32 IP address
MODEL_PATH = "tf_lite_model.tflite"
CONFIDENCE_THRESHOLD = 0.60  # Adjusted threshold to reduce false positives
EMAIL_SENDER = "vishwanth04adhavan@gmail.com"
EMAIL_RECEIVER = "sharavanavell@gmail.com"
EMAIL_PASSWORD = "yvtd nbsc ihxm issj"  # Use App Password if Gmail blocks access

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to send email
def send_email(image_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = "Accident Detected!"
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg.set_content("An accident has been detected. See the attached image.")
        
        with open(image_path, "rb") as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename="accident.jpg")
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

# Function to capture and test frames
def detect_accident():
    while True:
        try:
            response = requests.get(ESP32_URL, timeout=5)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Preprocess image
                resized_frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
                input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0
                
                # Run model
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Debugging: Print raw model output
                print("Raw Model Output:", output_data)
                
                # Adjust based on model output structure
                confidence = output_data[0][1] if output_data.shape[1] == 2 else output_data[0][0]
                prediction = "Accident" if confidence >= 0.5 else "No Accident"
                print(f"Detection Confidence: {confidence:.2f} | Prediction: {prediction}")
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    print("Accident detected! Sending email...")
                    img_path = "accident_detected.jpg"
                    cv2.imwrite(img_path, frame)
                    send_email(img_path)
                
            else:
                print("Failed to fetch frame from ESP32-CAM")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(5)  # Wait 5 seconds before checking again

# Start accident detection
detect_accident()
