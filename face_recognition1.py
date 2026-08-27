import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="😀",
    layout="centered"
)

st.title("😀 Face Recognition System")

# ==========================================================
# FOLDER
# ==========================================================

KNOWN_FOLDER = "known_faces"

os.makedirs(KNOWN_FOLDER, exist_ok=True)

# ==========================================================
# TABS
# ==========================================================

register_tab, recognize_tab = st.tabs(
    ["📝 Register Face", "🔍 Recognize Face"]
)

# ==========================================================
# REGISTER FACE
# ==========================================================

with register_tab:

    st.header("Register New Person")

    name = st.text_input(
        "Enter person's name"
    )

    uploaded_file = st.camera_input(
        "Take a photo"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Captured Image",
            use_container_width=True
        )

        if st.button("💾 Save Face"):

            if not name.strip():

                st.error("Please enter a name.")

            else:

                # Convert image to OpenCV format
                image_array = np.array(image)

                # RGB → BGR
                frame = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                # Face detector
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades +
                    "haarcascade_frontalface_default.xml"
                )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                if len(faces) == 0:

                    st.error(
                        "❌ No face detected. "
                        "Please take another photo."
                    )

                elif len(faces) > 1:

                    st.error(
                        "❌ Multiple faces detected. "
                        "Please capture only one person."
                    )

                else:

                    x, y, w, h = faces[0]

                    # Crop face
                    face = frame[
                        y:y+h,
                        x:x+w
                    ]

                    # Draw rectangle for display
                    display_image = frame.copy()

                    cv2.rectangle(
                        display_image,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        3
                    )

                    # Save face
                    filename = (
                        name.strip() + ".jpg"
                    )

                    file_path = os.path.join(
                        KNOWN_FOLDER,
                        filename
                    )

                    cv2.imwrite(
                        file_path,
                        face
                    )

                    st.success(
                        f"✅ {name} registered successfully!"
                    )

                    # Convert back RGB
                    display_image = cv2.cvtColor(
                        display_image,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        display_image,
                        caption="Detected Face",
                        use_container_width=True
                    )

# ==========================================================
# RECOGNIZE FACE
# ==========================================================

with recognize_tab:

    st.header("Recognize Person")

    test_image = st.camera_input(
        "Take a photo to recognize"
    )

    if test_image is not None:

        image = Image.open(test_image)

        image_array = np.array(image)

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # Face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:

            st.error("❌ No face detected.")

        else:

            for (x, y, w, h) in faces:

                face = frame[
                    y:y+h,
                    x:x+w
                ]

                person_name = "Unknown"

                # ------------------------------------------
                # Compare with registered faces
                # ------------------------------------------

                for filename in os.listdir(KNOWN_FOLDER):

                    if not filename.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    ):
                        continue

                    known_path = os.path.join(
                        KNOWN_FOLDER,
                        filename
                    )

                    try:

                        result = DeepFace.verify(
                            img1_path=known_path,
                            img2_path=face,
                            model_name="VGG-Face",
                            detector_backend="opencv",
                            enforce_detection=False
                        )

                        if result["verified"]:

                            person_name = os.path.splitext(
                                filename
                            )[0]

                            break

                    except Exception:
                        continue

                # ------------------------------------------
                # Draw box
                # ------------------------------------------

                if person_name == "Unknown":

                    box_color = (0, 0, 255)

                else:

                    box_color = (0, 255, 0)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    box_color,
                    3
                )

                cv2.putText(
                    frame,
                    person_name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    box_color,
                    2
                )

            # Convert BGR → RGB
            result_image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                result_image,
                caption="Recognition Result",
                use_container_width=True
            )

            if person_name == "Unknown":

                st.error("❌ Unknown Person")

            else:

                st.success(
                    f"✅ Known Person: {person_name}"
                )