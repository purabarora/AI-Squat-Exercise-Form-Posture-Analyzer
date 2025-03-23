# AI-Powered Squat Form Checker

This project is an interactive web application designed to analyze squat form and help prevent exercise-related injuries. It combines pose estimation with a custom-trained machine learning model to provide users with real-time feedback on their squat posture.

## Features

- **Pose Estimation:** Uses MediaPipe to detect key body landmarks.
- **Machine Learning Classifier:** Custom-trained MobileNetV2 model identifies rounded back posture.
- **Real-Time Feedback:** Offers actionable tips on knee alignment, torso lean, and overall squat form.
- **User-Friendly Interface:** Built with Streamlit for seamless image upload and feedback delivery.

## Technologies Used

- Python
- Streamlit
- TensorFlow & Keras
- OpenCV
- MediaPipe
- NumPy

## How to Run

1. Install dependencies:

   ```bash
   pip install streamlit tensorflow opencv-python mediapipe numpy
   ```

2. Run the application:

   ```bash
   streamlit run app.py
   ```

3. Upload your squat image (preferably a clear side-angle image at the deepest point of your squat) and receive instant feedback.

## Purpose

Squats are a fundamental exercise in strength training, but improper form can easily lead to knee strain, back issues, and other injuries. This project provides an AI-driven solution to promote proper technique and prevent injuries through real-time posture assessment.
