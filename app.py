import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('squat_classifier.h5')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

st.title("AI Squat Form Checker")

st.markdown("Tip: For the most accurate feedback, use a side-angle image of the deepest point in your squat.")
st.markdown("Remember: Your knees should never protrude inward during your squat.")

uploaded_file = st.file_uploader("Upload your squat image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption='Uploaded Image', use_container_width=True)

    pose = mp_pose.Pose()
    results = pose.process(img_rgb)

    annotated_img = img.copy()
    mp_drawing.draw_landmarks(annotated_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    detailed_feedback = ""
    pose_good = True

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def get_point(part):
            return [landmarks[mp_pose.PoseLandmark[part].value].x, landmarks[mp_pose.PoseLandmark[part].value].y]

        hip = get_point('RIGHT_HIP')
        knee = get_point('RIGHT_KNEE')
        ankle = get_point('RIGHT_ANKLE')
        shoulder = get_point('RIGHT_SHOULDER')

        knee_angle = calculate_angle(hip, knee, ankle)
        torso_angle = calculate_angle(shoulder, hip, knee)

        hip_to_ankle = euclidean_distance(hip, ankle)
        knee_to_ankle = euclidean_distance(knee, ankle)
        knee_ankle_ratio = knee_to_ankle / hip_to_ankle

        if knee_ankle_ratio > 1.05:
            detailed_feedback += "Tip: Keep your knees aligned with your toes. Avoid letting them move excessively forward.\n"
            pose_good = False
        else:
            detailed_feedback += "Great job keeping your knees in line with your toes.\n"

        if knee_angle > 160:
            detailed_feedback += "Tip: Try bending your knees more to properly engage muscles.\n"
            pose_good = False
        else:
            detailed_feedback += "Nice knee bend depth.\n"

        if torso_angle < 60:
            detailed_feedback += "Tip: Avoid leaning too far forward. Keep your chest up and back straight.\n"
            pose_good = False
        else:
            detailed_feedback += "Good upright posture maintained.\n"

        shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x

        if shoulder_x - hip_x > 0.05:
            detailed_feedback += "Tip: Your upper back appears rounded. Keep your chest up and shoulders back.\n"
            pose_good = False
        else:
            detailed_feedback += "Good back posture maintained.\n"

        nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x

        if nose_x - hip_x > 0.15:
            detailed_feedback += "Tip: Your head and neck appear too far forward. Keep head neutral.\n"
            pose_good = False
        else:
            detailed_feedback += "Good head and neck alignment maintained.\n"

    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    st.subheader("Overall Assessment:")
    final_good = pose_good and (prediction < 0.5)
    if final_good:
        st.success("Good squat form detected.")
    else:
        st.error("Please work on your squat form.")

    st.subheader("Detailed Feedback:")
    st.text(detailed_feedback)

    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption='Pose Estimation', use_container_width=True)
