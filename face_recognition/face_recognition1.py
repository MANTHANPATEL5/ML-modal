
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
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
# FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


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

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Captured Image",
            use_container_width=True
        )

        if st.button("💾 Save Face"):

            if not name.strip():

                st.error("Please enter a name.")

            else:

                # --------------------------------------------------
                # Convert PIL image to OpenCV format
                # --------------------------------------------------

                image_array = np.array(image)

                frame = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                # --------------------------------------------------
                # Convert to grayscale
                # --------------------------------------------------

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                # --------------------------------------------------
                # Detect faces
                # --------------------------------------------------

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                # --------------------------------------------------
                # No face
                # --------------------------------------------------

                if len(faces) == 0:

                    st.error(
                        "❌ No face detected. "
                        "Please take another photo."
                    )

                # --------------------------------------------------
                # Multiple faces
                # --------------------------------------------------

                elif len(faces) > 1:

                    st.error(
                        "❌ Multiple faces detected. "
                        "Please capture only one person."
                    )

                # --------------------------------------------------
                # One face
                # --------------------------------------------------

                else:

                    x, y, w, h = faces[0]

                    # Crop face
                    face = frame[
                        y:y + h,
                        x:x + w
                    ]

                    # --------------------------------------------------
                    # Draw rectangle
                    # --------------------------------------------------

                    display_image = frame.copy()

                    cv2.rectangle(
                        display_image,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

                    # --------------------------------------------------
                    # Create filename
                    # --------------------------------------------------

                    safe_name = name.strip()

                    filename = safe_name + ".jpg"

                    file_path = os.path.join(
                        KNOWN_FOLDER,
                        filename
                    )

                    # --------------------------------------------------
                    # Save cropped face
                    # --------------------------------------------------

                    success = cv2.imwrite(
                        file_path,
                        face
                    )

                    if success:

                        st.success(
                            f"✅ {safe_name} registered successfully!"
                        )

                        # Convert BGR → RGB
                        display_image = cv2.cvtColor(
                            display_image,
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            display_image,
                            caption="Detected Face",
                            use_container_width=True
                        )

                    else:

                        st.error(
                            "❌ Failed to save the face image."
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

        image = Image.open(test_image).convert("RGB")

        image_array = np.array(image)

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # --------------------------------------------------
        # Convert to grayscale
        # --------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # --------------------------------------------------
        # Detect faces
        # --------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # --------------------------------------------------
        # No face
        # --------------------------------------------------

        if len(faces) == 0:

            st.error("❌ No face detected.")

        # --------------------------------------------------
        # Multiple faces
        # --------------------------------------------------

        elif len(faces) > 1:

            st.error(
                "❌ Multiple faces detected. "
                "Please capture only one person."
            )

        # --------------------------------------------------
        # One face
        # --------------------------------------------------

        else:

            x, y, w, h = faces[0]

            # Crop detected face
            face = frame[
                y:y + h,
                x:x + w
            ]

            person_name = "Unknown"

            # --------------------------------------------------
            # Check registered faces
            # --------------------------------------------------

            known_files = os.listdir(KNOWN_FOLDER)

            image_files = [
                file for file in known_files
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            # --------------------------------------------------
            # No registered faces
            # --------------------------------------------------

            if len(image_files) == 0:

                st.warning(
                    "⚠️ No registered faces found. "
                    "Please register a person first."
                )

            else:

                # --------------------------------------------------
                # Compare with each registered face
                # --------------------------------------------------

                for filename in image_files:

                    known_path = os.path.join(
                        KNOWN_FOLDER,
                        filename
                    )

                    try:

                        result = DeepFace.verify(
                            img1_path=known_path,
                            img2_path=face,
                            model_name="VGG-Face",
                            detector_backend="skip",
                            enforce_detection=False
                        )

                        if result["verified"]:

                            person_name = os.path.splitext(
                                filename
                            )[0]

                            break

                    except Exception as e:

                        # Ignore individual comparison errors
                        continue

            # --------------------------------------------------
            # Select box color
            # --------------------------------------------------

            if person_name == "Unknown":

                box_color = (0, 0, 255)

            else:

                box_color = (0, 255, 0)

            # --------------------------------------------------
            # Draw rectangle
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                3
            )

            # --------------------------------------------------
            # Display person's name
            # --------------------------------------------------

            cv2.putText(
                frame,
                person_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                box_color,
                2
            )

            # --------------------------------------------------
            # Convert BGR → RGB
            # --------------------------------------------------

            result_image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                result_image,
                caption="Recognition Result",
                use_container_width=True
            )

            # --------------------------------------------------
            # Result message
            # --------------------------------------------------

            if person_name == "Unknown":

                st.error(
                    "❌ Unknown Person"
                )

            else:

                st.success(
                    f"✅ Known Person: {person_name}"
                )
