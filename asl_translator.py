import cv2
import mediapipe as mp
import math
import joblib
import os
import collections
import statistics
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Tracks the hand and gives us the 21 landmarks (x,y) for each hand
class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", num_hands=1):
        base_options = python.BaseOptions(model_asset_path=model_path) 
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self.detector.detect(mp_image)

# Extracts features from the hand landmarks to feed into the AI Brain
class FeatureExtractor:
    def extract(self, hand_landmarks, w, h):
        def to_pixel(landmark):
            return int(landmark.x * w), int(landmark.y * h)
        
        coords = [to_pixel(lm) for lm in hand_landmarks]

        wrist = coords[0]
        middle_mcp = coords[9]
        palm_size = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])

        if palm_size == 0: palm_size = 1 

        return {
            "coords": coords,
            "palm_size": palm_size,
            "raw": hand_landmarks 
        }

# Classifies gestures using the trained AI model
class GestureClassifier:
    def __init__(self):
        if os.path.exists("asl_model.pkl"):
            self.model = joblib.load("asl_model.pkl")
            print("Successfully loaded asl_model.pkl!")
        else:
            print("ERROR: asl_model.pkl not found!")
            self.model = None

        # Stores the last 10 predictions (Short Term Memory Buffer)
        self.prediction_history = collections.deque(maxlen=10)

    def predict(self, features):
        if self.model is None:
            return "No Model"

        coords = features["coords"]
        wrist = coords[0]
        palm_size = features["palm_size"]

        # Flatten and normalize the coordinates relative to the wrist and scaled by palm size
        feature_vector = []
        for (x, y) in coords:
            norm_x = (x - wrist[0]) / palm_size
            norm_y = (y - wrist[1]) / palm_size
            feature_vector.extend([norm_x, norm_y])
        
        # Get the raw prediction from the model
        raw_prediction = self.model.predict([feature_vector])[0]
        
        # Add the new guess to the memory buffer
        self.prediction_history.append(raw_prediction)
        
        # Find the most common guess in the buffer (the majority vote)
        smoothed_prediction = statistics.mode(self.prediction_history)
        
        return smoothed_prediction

# Main App
class ASLApp:
    def __init__(self):
        self.tracker = HandTracker()
        self.extractor = FeatureExtractor()
        self.classifier = GestureClassifier()
        
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
        ]

    def draw_skeleton(self, frame, coords):
        for connection in self.HAND_CONNECTIONS:
            cv2.line(frame, coords[connection[0]], coords[connection[1]], (255, 0, 0), 2)
        for point in coords:
            cv2.circle(frame, point, 6, (0, 255, 0), -1)
            cv2.circle(frame, point, 3, (255, 255, 0), -1)

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.tracker.detect(rgb_frame)

            if results.hand_landmarks:
                # We only grab the first hand detected to keep it simple
                hand_landmarks = results.hand_landmarks[0]
                
                # Extract and Predict
                features = self.extractor.extract(hand_landmarks, w, h)
                letter = self.classifier.predict(features)

                # Draw UI
                self.draw_skeleton(frame, features["coords"])
                
                # Only draw the text if the letter is NOT our "Nothing" class (0)
                if letter and letter != "0":
                    cv2.putText(frame, f"Letter: {letter}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

            cv2.imshow('ML Sign Language Translator', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ASLApp()
    app.run()