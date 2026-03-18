import cv2
import mediapipe as mp
import math
import os
import csv
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

# Main App
class ASLApp:
    def __init__(self):
        self.tracker = HandTracker()
        self.extractor = FeatureExtractor()
        
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
            (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
        ]
        
        # Dataset collection variables
        self.dataset_file = "asl_dataset.csv"
        self.current_label = "A"
        self.samples_collected = 0

        # Create the CSV header if the file doesn't exist
        if not os.path.exists(self.dataset_file):
            with open(self.dataset_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Header: Label, then x0, y0, x1, y1... x20, y20
                header = ["label"]
                for i in range(21):
                    header.extend([f"x{i}", f"y{i}"])
                writer.writerow(header)

    def log_data(self, features, label):
        """Flattens the hand coordinates, normalizes them, and saves to CSV"""
        coords = features["coords"]
        wrist = coords[0]
        palm_size = features["palm_size"]
        
        row = [label]
        for (x, y) in coords:
            # Normalize: Make wrist (0,0) and scale by palm size
            norm_x = (x - wrist[0]) / palm_size
            norm_y = (y - wrist[1]) / palm_size
            row.extend([norm_x, norm_y])
            
        with open(self.dataset_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            self.samples_collected += 1

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
            
            # Check keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # If user presses a letter a-z OR a number 0-9
            elif ord('a') <= key <= ord('z') or ord('0') <= key <= ord('9'):
                self.current_label = chr(key).upper()
                self.samples_collected = 0

            # Check if spacebar is being held down (ASCII 32)
            is_recording = (key == 32) 

            if results.hand_landmarks:
                for i, hand_landmarks in enumerate(results.hand_landmarks):
                    hand_label = results.handedness[i][0].category_name
                    features = self.extractor.extract(hand_landmarks, w, h)
                    
                    self.draw_skeleton(frame, features["coords"])
                    
                    # Record data if spacebar is held down
                    if is_recording:
                        self.log_data(features, self.current_label)
                        cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)

            ui_text = f"Target: {self.current_label} | Collected: {self.samples_collected}"
            cv2.putText(frame, ui_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Press A-Z to change target. Hold SPACE to record.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('ASL Data Collector', frame)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ASLApp()
    app.run()