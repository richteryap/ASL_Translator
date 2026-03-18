# ML-Powered ASL Translator

> A real-time American Sign Language (ASL) translator built with Python, OpenCV, MediaPipe, and Scikit-Learn.

This project uses a modular Machine Learning pipeline to extract 3D hand landmarks, normalize the mathematical data for scale-invariance (palm-size normalization), and classify the static hand signs using a Random Forest algorithm.

---

## Features

* **Real-Time Tracking:** Utilizes Google's MediaPipe Tasks API for lightning-fast hand tracking.
* **Scale-Invariant Math:** Calculates relative distances based on palm size, meaning the AI recognizes your signs regardless of how close or far your hand is from the camera.
* **Temporal Smoothing:** Features a 10-frame sliding window (Memory Buffer) to eliminate UI flickering and stabilize live predictions.
* **Null-Class Detection:** Includes a "Background" class (0) to actively ignore open palms and resting hands, preventing false positive letter detections.

---

## Project Structure

```text
ASL_Translator
 ┣ collect_data.py      # The Camera: Records normalized hand coordinates to CSV
 ┣ train_model.py       # The Brain Builder: Trains the Random Forest AI
 ┣ asl_translator.py    # The Final App: Live webcam ASL translation
 ┣ requirements.txt     # Dependencies
 ┣ .gitignore           # Keeps the repo clean of large binaries/data
 ┗ README.md
```

## Installation
**1. Clone the repository:**
```text
git clone [https://github.com/richteryap/ASL_Translator.git](https://github.com/richteryap/ASL_Translator.git)
cd ASL_Translator
```
**2. Create a virtual environment (Recommended):**
```text
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# source .venv/bin/activate    # On Mac/Linux
```
**3. Install the dependencies:**
```text
pip install -r requirements.txt
```

## How to Use
This project is designed to be highly customizable. You must collect your own hand data or email me for the dataset to train the model before translating.

**Step 1: Collect Data (collect_data.py)**
Run this script to build your dataset.

1. Run python collect_data.py.

2. Press 0 on your keyboard to set the target to the background class. Hold **SPACEBAR** while moving an open palm or resting hand around the screen to record "noise" (Aim for 200 samples).

3. Press A through Z to change the target letter. Make the corresponding ASL sign, and hold **SPACEBAR** to record samples for each letter.

4. Close the app. A file named asl_dataset.csv will be generated.

**Step 2: Train the AI (train_model.py)**
Run this script to teach the AI the patterns you just recorded.

1. Run python train_model.py.

2. The script will automatically split your data, train a Random Forest Classifier, and grade itself on a hidden test set.

3. Upon success, it will save a compiled brain file named asl_model.pkl.

**Step 3: Live Translation (asl_translator.py)**
Run the final application.

1. Run python asl_translator.py.

2. Step back and sign! The app will seamlessly translate your signs in real-time. If you open your hand or drop it, the text will safely disappear thanks to the background class.

## Future Roadmap
* **Add support for dynamic/motion-based signs (J, Z).**