
import cv2
import requests
import time
import serial
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from datetime import datetime

# Configuration
IP_WEBCAM_URL = "http://192.168.18.196:8080/shot.jpg"  # Replace with your IP Webcam URL
ARDUINO_PORT = "COM5"  # Replace actual COM port
ARDUINO_BAUD_RATE = 9600
MODEL_PATH = "C:\\Users\\91702\\Downloads\\2022-03-18_food_not_food_model_efficientnet_lite0_v1.tflite"  
INPUT_SIZE = (224, 224)  # Standard input size for most models

# Folder to save food images
FOOD_IMAGES_FOLDER = "C:\\pyhtion\\saved_images"

# Classes for the model
class_names = ["food", "not_food"]

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def save_food_image(image):
    """Save food image with timestamp to the food images folder."""
    # Ensure the food images folder exists
    folder = ensure_directory_exists(FOOD_IMAGES_FOLDER)
    
    # Create a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"food_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)
    
    # Save the image
    cv2.imwrite(filepath, image)
    print(f"Food image saved: {filepath}")
    return filepath

def load_tflite_model():
    """Load the TFLite model."""
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("TFLite model loaded successfully")
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None, None, None

def connect_to_arduino():
    """Connect to the Arduino."""
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD_RATE, timeout=1)
        print("Connected to Arduino")
        time.sleep(2)  # Allow time for Arduino to reset
        return arduino
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino: {e}")
        return None

def capture_image_from_ip_webcam():
    """Capture an image from the IP Webcam."""
    try:
        response = requests.get(IP_WEBCAM_URL)
        if response.status_code == 200:
            # Convert the image from bytes to a numpy array
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"Failed to get image from IP Webcam. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def preprocess_image(image):
    """Preprocess the image for the TFLite model."""
    # Resize the image first
    resized = cv2.resize(image, INPUT_SIZE)
    
    # Convert from BGR (OpenCV format) to RGB
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Keep as UINT8 (don't convert to float)
    # Add batch dimension
    img_array = np.expand_dims(image_rgb, axis=0)
    
    return img_array

def is_food(image, interpreter, input_details, output_details):
    """Check if the image contains food using the TFLite model."""
    try:
        # Preprocess the image
        img_array = preprocess_image(image)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Print raw output data for debugging
        print(f"Raw model output: {output_data}")
        
        # Process the results
        if len(output_data[0]) == 2:  # Binary classification (food, not_food)
            food_probability = output_data[0][0]  # First value is food probability
            not_food_probability = output_data[0][1]  # Second value is not-food probability
            
            print(f"Food score: {food_probability:.4f}, Not-food score: {not_food_probability:.4f}")
            
            # Use direct probability comparison instead of argmax
            is_food_detected = food_probability > not_food_probability
            print(f"Detection result: {'FOOD' if is_food_detected else 'NOT FOOD'}")
            
            # Try with reversed logic as a debug test - the model might have swapped classes
            reversed_logic = not_food_probability < food_probability
            if reversed_logic != is_food_detected:
                print("Warning: Logic inconsistency in food detection")
            
            return is_food_detected
        else:
            # If the output format is different, adjust accordingly
            print(f"Unexpected output format: {output_data}")
            return False
    except Exception as e:
        print(f"Error in food detection: {e}")
        return False

def control_bins(arduino, is_food_item):
    """Control the bins based on food detection."""
    if arduino:
        if is_food_item:
            print("Food detected! Opening RED bin...")
            arduino.write(b'R')  # Send 'R' to Arduino to open red bin
        else:
            print("Not food! Opening GREEN bin...")
            arduino.write(b'G')  # Send 'G' to Arduino to open green bin
        
        # Wait for confirmation from Arduino (optional)
        response = arduino.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")
    else:
        # Arduino simulation mode
        if is_food_item:
            print("SIMULATION: Food detected! Opening RED bin...")
        else:
            print("SIMULATION: Not food! Opening GREEN bin...")

def test_with_known_food():
    """Test the model with a known food image."""
    # Load a known food image
    test_img_path = "path/to/known_food_image.jpg"  # Replace with an actual path
    try:
        test_img = cv2.imread(test_img_path)
        if test_img is None:
            print(f"Failed to load test image: {test_img_path}")
            return
            
        # Load model
        interpreter, input_details, output_details = load_tflite_model()
        if interpreter is None:
            print("Model loading failed")
            return
            
        # Test detection
        result = is_food(test_img, interpreter, input_details, output_details)
        print(f"Test image result: {'FOOD' if result else 'NOT FOOD'}")
    except Exception as e:
        print(f"Error during test: {e}")

def main():
    # Ensure the food images folder exists
    ensure_directory_exists(FOOD_IMAGES_FOLDER)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
        
    # Load the TFLite model
    interpreter, input_details, output_details = load_tflite_model()
    if interpreter is None:
        print("Exiting due to model loading failure")
        return
    
    # Connect to Arduino (now optional)
    arduino = connect_to_arduino()
    if not arduino:
        print("Arduino not connected. Running in simulation mode.")
    
    # Try alternative class order for better results
    print("Current class order:", class_names)
    alternative_classes = ["not_food", "food"]  # Try reversed order
    print("Alternative class order (for debugging):", alternative_classes)
    
    try:
        # Try to use webcam if IP webcam fails
        use_webcam = False
        webcam_cap = None
        
        while True:
            print("\nCapturing new image...")
            
            # Capture image
            image = None
            if not use_webcam:
                # Try IP webcam first
                image = capture_image_from_ip_webcam()
                if image is None:
                    print("IP Webcam failed. Switching to local webcam.")
                    use_webcam = True
                    webcam_cap = cv2.VideoCapture(0)
            
            if use_webcam:
                if webcam_cap is None:
                    webcam_cap = cv2.VideoCapture(0)
                ret, image = webcam_cap.read()
                if not ret:
                    print("Failed to capture from webcam. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
            
            if image is None:
                print("Failed to capture image. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Display the captured image
            cv2.imshow("Captured Image", image)
            key = cv2.waitKey(1)  # Update the window
            
            # Check if the image contains food
            food_detected = is_food(image, interpreter, input_details, output_details)
            
            # Save image if it contains food
            if food_detected:
                saved_path = save_food_image(image)
                print(f"Food image saved to: {saved_path}")
            
            # Control the bins
            control_bins(arduino, food_detected)
            
            # Handle exit key
            if key & 0xFF == ord('q'):
                print("Exiting program...")
                break
            
            # Wait before the next detection
            print("Waiting for 5 seconds before next detection...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        if arduino:
            arduino.close()
        if 'webcam_cap' in locals() and webcam_cap is not None:
            webcam_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
