# ML-Powered ASL Translator

A real-time American Sign Language (ASL) translator built with Python, OpenCV, MediaPipe, and Scikit-Learn. 

This project uses a modular Machine Learning pipeline to extract 3D hand landmarks, normalize the mathematical data for scale-invariance (palm-size normalization), and classify the static hand signs using a Random Forest algorithm.

## Features
* **Real-Time Tracking:** Utilizes Google's MediaPipe Tasks API for lightning-fast hand tracking.
* **Scale-Invariant Math:** Calculates relative distances based on palm size, meaning the AI recognizes your signs regardless of how close or far your hand is from the camera.
* **Temporal Smoothing:** Features a 10-frame sliding window (Memory Buffer) to eliminate UI flickering and stabilize live predictions.
* **Null-Class Detection:** Includes a "Background" class (0) to actively ignore open palms and resting hands, preventing false positive letter detections.

## Project Structure
```text
asl-ml-translator
 ┣ collect_data.py      # The Camera: Records normalized hand coordinates to CSV
 ┣ train_model.py       # The Brain Builder: Trains the Random Forest AI
 ┣ asl_translator.py    # The Final App: Live webcam ASL translation
 ┣ requirements.txt     # Dependencies
 ┣ .gitignore           # Keeps the repo clean of large binaries/data
 ┗ README.md
