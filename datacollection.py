import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model/model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe setup for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Confidence threshold (e.g., 0.7 means 70% confidence required)
CONFIDENCE_THRESHOLD = 0.7

# Function to preprocess the hand image
def preprocess_hand_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect hands in the image using MediaPipe
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to RGB before processing

    if results.multi_hand_landmarks:
        # For now, we are considering the first hand detected (you can customize this)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Crop around the hand (you can refine this based on the landmarks)
        x_min, y_min, x_max, y_max = 9999, 9999, 0, 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            x_min, y_min = min(x, x_min), min(y, y_min)
            x_max, y_max = max(x, x_max), max(y, y_max)

        # Crop the image to the bounding box of the hand
        cropped_hand = gray_image[y_min:y_max, x_min:x_max]

        # Resize the cropped hand region to 224x224 (instead of 64x64)
        cropped_hand_resized = cv2.resize(cropped_hand, (224, 224))

        # Normalize the pixel values to [0, 1]
        cropped_hand_resized = cropped_hand_resized / 255.0

        # Convert grayscale to 3 channels (RGB)
        cropped_hand_resized = np.stack([cropped_hand_resized] * 3, axis=-1)  # Stack to create 3 channels

        return cropped_hand_resized
    else:
        return None  # No hand detected in the image

# Function to make predictions on the input image
def predict_sign(frame):
    processed_img = preprocess_hand_image(frame)

    if processed_img is not None:
        # Expand dimensions to fit the model input
        input_data = np.expand_dims(processed_img, axis=0)

        # Set the tensor for the model input
        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

        # Run the model
        interpreter.invoke()

        # Get the output prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)  # Get the class with the highest score
        confidence = np.max(output_data)  # Get the confidence score for the prediction

        # Map numeric prediction (0, 1) to corresponding sign language labels
        labels = {0: 'good', 1: 'hi'}  # Corrected mapping: 0 -> 'good', 1 -> 'hi'

        # Only return the prediction if it meets the confidence threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            return labels.get(prediction, 'Unknown')  # Return the corresponding label
        else:
            return None  # If confidence is below the threshold, return None
    else:
        return None

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Store previous signs in a list
previous_signs = []
last_sign = None  # Track the last predicted sign
last_prediction_time = None  # Track the time of the last prediction

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame was captured
    if not ret:
        break

    # Make a prediction
    sign = predict_sign(frame)

    # Ensure the sign prediction is only added after some time has passed (1-2 seconds)
    if sign is not None and sign != last_sign:
        # Get the current time
        current_time = time.time()

        # Only update the sign after 1 second
        if last_prediction_time is None or (current_time - last_prediction_time) > 1:
            # Add the new sign to the list and update the last_sign
            previous_signs.append(sign)
            last_sign = sign
            last_prediction_time = current_time

    # Create a smaller white space at the bottom for displaying previous signs
    history_area_height = 50  # Adjust the height to make the space smaller
    cv2.rectangle(frame, (0, frame.shape[0] - history_area_height), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)

    # Join the previous signs into a single string, separating by a space
    history_text = "   ".join(previous_signs[-5:])  # Show the last 5 signs

    # Display the history of signs horizontally in the bottom space
    cv2.putText(frame, history_text, (10, frame.shape[0] - history_area_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()


