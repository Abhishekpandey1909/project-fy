import cv2                                                       #type: ignore
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Load the model and labels
model = load_model("E:\\yoga dataet\\code\\yoga_pose_neural_network_model.h5")
LABELS = ["downdog", "goddess", "plank", "tree", "warrior2"]





import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Load the model and labels
model = load_model("E:\\yoga dataet\\code\\yoga_pose_neural_network_model.h5")
LABELS = ["downdog", "goddess", "plank", "tree", "warrior2"]

# Streamlit app title

# Streamlit app title
st.title("YOGGI - Yoga Pose Detection App")
st.subheader("B.Tech Major Project under the guidance of Prof. Shivesh Sharma")


# Sidebar for pose selection
chosen_pose = st.sidebar.selectbox("Choose the pose you want to perform:", LABELS)
st.sidebar.write(f"You selected: {chosen_pose.capitalize()}")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to extract landmarks
def extract_landmarks(results):
    """Extract pose landmarks as a flattened list."""
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(132)  # 33 landmarks * 4 attributes

# Use the webcam
use_webcam = st.checkbox("Use Webcam")

if use_webcam:
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("No frame detected.")
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose.process(image)

        # Draw pose landmarks on the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Extract landmarks
        landmarks = extract_landmarks(results)

        if landmarks.sum() != 0:
            landmarks = landmarks.reshape(1, -1)
            pose_class = model.predict(landmarks)
            pose_class_index = np.argmax(pose_class)
            predicted_pose = LABELS[pose_class_index]

            pose_probability = pose_class[0][pose_class_index] * 100  # Probability of the predicted pose

            # Add feedback on the frame
            feedback = ""
            if pose_probability < 80 or predicted_pose != chosen_pose:
                feedback = "No pose detected or incorrect pose."
                color = (255, 0, 0)
            else:
                feedback = "Good alignment!"
                color = (0, 255, 0)

            cv2.putText(
                annotated_image,
                f"Pose: {predicted_pose} ({pose_probability:.2f}%)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_image,
                feedback,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                annotated_image,
                "No pose detected.",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Display the frame
        stframe.image(annotated_image, channels="RGB")

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Check 'Use Webcam' to start.")
