import os

# ==========================================================
# ENVIRONMENT SETTINGS
# ==========================================================

# Set before importing TensorFlow / DeepFace
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# ==========================================================
# STREAMLIT
# ==========================================================

import streamlit as st


# ==========================================================
# OPENCV TEST
# ==========================================================

# Import cv2 separately so Streamlit can show the real error
try:
    import cv2
except Exception as e:
    st.error("❌ OpenCV (cv2) could not be loaded.")
    st.code(
        f"{type(e).__name__}: {e}",
        language="text"
    )

    st.warning(
        "Check requirements.txt and packages.txt in the "
        "face_recognition folder, then redeploy the app."
    )

    st.stop()


# ==========================================================
# OTHER IMPORTS
# ==========================================================

import numpy as np
from PIL import Image


# ==========================================================
# DEEPFACE TEST
# ==========================================================

try:
    from deepface import DeepFace
except Exception as e:
    st.error("❌ DeepFace could not be loaded.")
    st.code(
        f"{type(e).__name__}: {e}",
        language="text"
    )
    st.stop()


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="😀",
    layout="centered"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("😀 Face Recognition System")

st.write(
    "Register a person and recognize them using the camera."
)


# ==========================================================
# OPENCV INFORMATION
# ==========================================================

# Show OpenCV version only if everything loaded correctly
st.caption(
    f"OpenCV: {cv2.__version__}"
)


# ==========================================================
# KNOWN FACES FOLDER
# ==========================================================

KNOWN_FOLDER = "known_faces"

os.makedirs(
    KNOWN_FOLDER,
    exist_ok=True
)


# ==========================================================
# FACE DETECTOR
# ==========================================================

cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    cascade_path
)

if face_cascade.empty():

    st.error(
        "❌ OpenCV face detector could not be loaded."
    )

    st.code(
        cascade_path,
        language="text"
    )

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

    # ------------------------------------------------------
    # NAME
    # ------------------------------------------------------

    name = st.text_input(
        "Enter person's name",
        key="register_name"
    )

    # ------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------

    uploaded_file = st.camera_input(
        "Take a photo",
        key="register_camera"
    )

    # ------------------------------------------------------
    # IMAGE CAPTURED
    # ------------------------------------------------------

    if uploaded_file is not None:

        try:

            # Read image
            image = Image.open(
                uploaded_file
            ).convert("RGB")

            # Display captured image
            st.image(
                image,
                caption="Captured Image"
            )

            # --------------------------------------------------
            # SAVE BUTTON
            # --------------------------------------------------

            if st.button(
                "💾 Save Face",
                key="save_face_button"
            ):

                # ------------------------------------------------
                # CHECK NAME
                # ------------------------------------------------

                if not name.strip():

                    st.error(
                        "❌ Please enter a name."
                    )

                else:

                    # --------------------------------------------
                    # CLEAN NAME
                    # --------------------------------------------

                    safe_name = name.strip()

                    safe_name = "".join(
                        character
                        for character in safe_name
                        if (
                            character.isalnum()
                            or character in (" ", "_", "-")
                        )
                    )

                    safe_name = safe_name.strip()

                    if not safe_name:

                        st.error(
                            "❌ Please enter a valid name."
                        )

                    else:

                        # ----------------------------------------
                        # PIL → NUMPY
                        # ----------------------------------------

                        image_array = np.array(
                            image
                        )

                        # ----------------------------------------
                        # RGB → BGR
                        # ----------------------------------------

                        frame = cv2.cvtColor(
                            image_array,
                            cv2.COLOR_RGB2BGR
                        )

                        # ----------------------------------------
                        # BGR → GRAYSCALE
                        # ----------------------------------------

                        gray = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2GRAY
                        )

                        # ----------------------------------------
                        # FACE DETECTION
                        # ----------------------------------------

                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(80, 80)
                        )

                        # ----------------------------------------
                        # NO FACE
                        # ----------------------------------------

                        if len(faces) == 0:

                            st.error(
                                "❌ No face detected. "
                                "Please take another photo."
                            )

                        # ----------------------------------------
                        # MULTIPLE FACES
                        # ----------------------------------------

                        elif len(faces) > 1:

                            st.error(
                                "❌ Multiple faces detected. "
                                "Please capture only one person."
                            )

                        # ----------------------------------------
                        # ONE FACE
                        # ----------------------------------------

                        else:

                            x, y, w, h = faces[0]

                            # ------------------------------------
                            # CROP FACE
                            # ------------------------------------

                            face = frame[
                                y:y + h,
                                x:x + w
                            ]

                            # ------------------------------------
                            # CHECK FACE SIZE
                            # ------------------------------------

                            if face.size == 0:

                                st.error(
                                    "❌ Could not crop the face."
                                )

                            else:

                                # -------------------------------
                                # DRAW RECTANGLE
                                # -------------------------------

                                display_image = frame.copy()

                                cv2.rectangle(
                                    display_image,
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0),
                                    3
                                )

                                # -------------------------------
                                # FILE NAME
                                # -------------------------------

                                filename = (
                                    safe_name +
                                    ".jpg"
                                )

                                file_path = os.path.join(
                                    KNOWN_FOLDER,
                                    filename
                                )

                                # -------------------------------
                                # SAVE FACE
                                # -------------------------------

                                success = cv2.imwrite(
                                    file_path,
                                    face
                                )

                                if success:

                                    st.success(
                                        f"✅ {safe_name} "
                                        "registered successfully!"
                                    )

                                    # ---------------------------
                                    # BGR → RGB
                                    # ---------------------------

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
                                        "❌ Failed to save "
                                        "the face image."
                                    )

        except Exception as e:

            st.error(
                "❌ Error while processing the "
                "registration image."
            )

            st.code(
                f"{type(e).__name__}: {e}",
                language="text"
            )


# ==========================================================
# RECOGNIZE FACE
# ==========================================================

with recognize_tab:

    st.header("Recognize Person")

    # ------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------

    test_image = st.camera_input(
        "Take a photo to recognize",
        key="recognize_camera"
    )

    # ------------------------------------------------------
    # IMAGE CAPTURED
    # ------------------------------------------------------

    if test_image is not None:

        try:

            # -----------------------------------------------
            # READ IMAGE
            # -----------------------------------------------

            image = Image.open(
                test_image
            ).convert("RGB")

            # -----------------------------------------------
            # PIL → NUMPY
            # -----------------------------------------------

            image_array = np.array(
                image
            )

            # -----------------------------------------------
            # RGB → BGR
            # -----------------------------------------------

            frame = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

            # -----------------------------------------------
            # BGR → GRAYSCALE
            # -----------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # -----------------------------------------------
            # FACE DETECTION
            # -----------------------------------------------

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            # -----------------------------------------------
            # NO FACE
            # -----------------------------------------------

            if len(faces) == 0:

                st.error(
                    "❌ No face detected."
                )

            # -----------------------------------------------
            # MULTIPLE FACES
            # -----------------------------------------------

            elif len(faces) > 1:

                st.error(
                    "❌ Multiple faces detected. "
                    "Please capture only one person."
                )

            # -----------------------------------------------
            # ONE FACE
            # -----------------------------------------------

            else:

                x, y, w, h = faces[0]

                # -------------------------------------------
                # CROP FACE
                # -------------------------------------------

                face = frame[
                    y:y + h,
                    x:x + w
                ]

                person_name = "Unknown"

                # -------------------------------------------
                # GET KNOWN FILES
                # -------------------------------------------

                try:

                    known_files = os.listdir(
                        KNOWN_FOLDER
                    )

                except Exception:

                    known_files = []

                # -------------------------------------------
                # IMAGE FILES
                # -------------------------------------------

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

                # -------------------------------------------
                # NO REGISTERED FACES
                # -------------------------------------------

                if len(image_files) == 0:

                    st.warning(
                        "⚠️ No registered faces found. "
                        "Please register a person first."
                    )

                # -------------------------------------------
                # COMPARE
                # -------------------------------------------

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

                            # ------------------------------
                            # CHECK RESULT
                            # ------------------------------

                            if result.get(
                                "verified",
                                False
                            ):

                                person_name = (
                                    os.path.splitext(
                                        filename
                                    )[0]
                                )

                                break

                        except Exception:

                            # Continue with next registered face
                            continue

                # -------------------------------------------
                # COLOR
                # -------------------------------------------

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

                # -------------------------------------------
                # DRAW FACE RECTANGLE
                # -------------------------------------------

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    3
                )

                # -------------------------------------------
                # TEXT POSITION
                # -------------------------------------------

                text_y = y - 10

                if text_y < 30:

                    text_y = y + h + 30

                # -------------------------------------------
                # DRAW NAME
                # -------------------------------------------

                cv2.putText(
                    frame,
                    person_name,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    box_color,
                    2
                )

                # -------------------------------------------
                # BGR → RGB
                # -------------------------------------------

                result_image = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # -------------------------------------------
                # DISPLAY RESULT
                # -------------------------------------------

                st.image(
                    result_image,
                    caption="Recognition Result"
                )

                # -------------------------------------------
                # RESULT MESSAGE
                # -------------------------------------------

                if person_name == "Unknown":

                    st.error(
                        "❌ Unknown Person"
                    )

                else:

                    st.success(
                        f"✅ Known Person: {person_name}"
                    )

        except Exception as e:

            st.error(
                "❌ Error while processing "
                "the recognition image."
            )

            st.code(
                f"{type(e).__name__}: {e}",
                language="text"
            )
