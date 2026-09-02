import os

# Must be before TensorFlow / DeepFace imports
os.environ["TF_USE_LEGACY_KERAS"] = "0"

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

st.write(
    "Register a person and recognize them using the camera."
)


# ==========================================================
# KNOWN FACES FOLDER
# ==========================================================

KNOWN_FOLDER = "known_faces"

os.makedirs(KNOWN_FOLDER, exist_ok=True)


# ==========================================================
# OPENCV FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("❌ OpenCV face detector could not be loaded.")
    st.stop()


# ==========================================================
# TABS
# ==========================================================

register_tab, recognize_tab = st.tabs(
    [
        "📝 Register Face",
        "🔍 Recognize Face"
    ]
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

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Captured Image"
        )

        save_button = st.button(
            "💾 Save Face"
        )

        if save_button:

            if not name.strip():

                st.error(
                    "❌ Please enter a name."
                )

            else:

                # ==================================================
                # CONVERT IMAGE
                # ==================================================

                image_array = np.array(image)

                frame = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                # ==================================================
                # GRAYSCALE
                # ==================================================

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                # ==================================================
                # DETECT FACE
                # ==================================================

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                # ==================================================
                # NO FACE
                # ==================================================

                if len(faces) == 0:

                    st.error(
                        "❌ No face detected. "
                        "Please take another photo."
                    )

                # ==================================================
                # MULTIPLE FACES
                # ==================================================

                elif len(faces) > 1:

                    st.error(
                        "❌ Multiple faces detected. "
                        "Please capture only one person."
                    )

                # ==================================================
                # ONE FACE
                # ==================================================

                else:

                    x, y, w, h = faces[0]

                    # ==================================================
                    # CROP FACE
                    # ==================================================

                    face = frame[
                        y:y + h,
                        x:x + w
                    ]

                    # ==================================================
                    # DRAW RECTANGLE
                    # ==================================================

                    display_image = frame.copy()

                    cv2.rectangle(
                        display_image,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

                    # ==================================================
                    # CLEAN NAME
                    # ==================================================

                    safe_name = name.strip()

                    safe_name = "".join(
                        character
                        for character in safe_name
                        if character.isalnum()
                        or character in (" ", "_", "-")
                    )

                    safe_name = safe_name.strip()

                    if not safe_name:

                        st.error(
                            "❌ Please enter a valid name."
                        )

                    else:

                        filename = safe_name + ".jpg"

                        file_path = os.path.join(
                            KNOWN_FOLDER,
                            filename
                        )

                        # ==================================================
                        # SAVE FACE
                        # ==================================================

                        success = cv2.imwrite(
                            file_path,
                            face
                        )

                        if success:

                            st.success(
                                f"✅ {safe_name} registered successfully!"
                            )

                            display_image = cv2.cvtColor(
                                display_image,
                                cv2.COLOR_BGR2RGB
                            )

                            st.image(
                                display_image,
                                caption="Detected Face"
                            )

                        else:

                            st.error(
                                "❌ Failed to save face image."
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

        image = Image.open(
            test_image
        ).convert("RGB")

        image_array = np.array(image)

        # ==================================================
        # RGB → BGR
        # ==================================================

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        # ==================================================
        # GRAYSCALE
        # ==================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================================
        # DETECT FACE
        # ==================================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # ==================================================
        # NO FACE
        # ==================================================

        if len(faces) == 0:

            st.error(
                "❌ No face detected."
            )

        # ==================================================
        # MULTIPLE FACES
        # ==================================================

        elif len(faces) > 1:

            st.error(
                "❌ Multiple faces detected. "
                "Please capture only one person."
            )

        # ==================================================
        # ONE FACE
        # ==================================================

        else:

            x, y, w, h = faces[0]

            # ==================================================
            # CROP FACE
            # ==================================================

            face = frame[
                y:y + h,
                x:x + w
            ]

            person_name = "Unknown"

            # ==================================================
            # GET REGISTERED FACES
            # ==================================================

            known_files = os.listdir(
                KNOWN_FOLDER
            )

            image_files = [
                file
                for file in known_files
                if file.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            ]

            # ==================================================
            # NO REGISTERED FACES
            # ==================================================

            if len(image_files) == 0:

                st.warning(
                    "⚠️ No registered faces found. "
                    "Please register a person first."
                )

            # ==================================================
            # COMPARE FACES
            # ==================================================

            else:

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

                        if result.get(
                            "verified",
                            False
                        ):

                            person_name = os.path.splitext(
                                filename
                            )[0]

                            break

                    except Exception:

                        continue

            # ==================================================
            # BOX COLOR
            # ==================================================

            if person_name == "Unknown":

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )

            # ==================================================
            # DRAW RECTANGLE
            # ==================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                3
            )

            # ==================================================
            # DISPLAY NAME
            # ==================================================

            text_y = y - 10

            if text_y < 30:

                text_y = y + h + 30

            cv2.putText(
                frame,
                person_name,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                box_color,
                2
            )

            # ==================================================
            # BGR → RGB
            # ==================================================

            result_image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ==================================================
            # DISPLAY RESULT
            # ==================================================

            st.image(
                result_image,
                caption="Recognition Result"
            )

            # ==================================================
            # RESULT
            # ==================================================

            if person_name == "Unknown":

                st.error(
                    "❌ Unknown Person"
                )

            else:

                st.success(
                    f"✅ Known Person: {person_name}"
                )
