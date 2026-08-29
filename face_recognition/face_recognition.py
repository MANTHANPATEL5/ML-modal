import cv2
import os
from deepface import DeepFace

# ==========================================================
# SETTINGS
# ==========================================================

KNOWN_FOLDER = "known_faces"

# ==========================================================
# CHECK KNOWN FACES
# ==========================================================

if not os.path.exists(KNOWN_FOLDER):
    print("❌ known_faces folder not found.")
    print("First run: python 01_register_face.py")
    exit()

known_images = [
    os.path.join(KNOWN_FOLDER, file)
    for file in os.listdir(KNOWN_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(known_images) == 0:
    print("❌ No registered faces found.")
    print("First run: python 01_register_face.py")
    exit()

print("✅ Registered faces:")

for image in known_images:
    print("   ", os.path.basename(image))

# ==========================================================
# FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================================
# OPEN CAMERA
# ==========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera not found")
    exit()

print("Camera started.")
print("Press Q to quit.")

frame_count = 0

# Store previous result
results = {}

while True:

    ret, frame = camera.read()

    if not ret:
        print("❌ Could not read camera")
        break

    frame_count += 1

    # ======================================================
    # DETECT FACES
    # ======================================================

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

    # ======================================================
    # RECOGNITION
    # ======================================================

    if frame_count % 10 == 0:

        results = {}

        for i, (x, y, w, h) in enumerate(faces):

            # Crop detected face
            face = frame[y:y+h, x:x+w]

            person_name = "UNKNOWN"

            # Compare with all registered faces
            for known_image in known_images:

                try:

                    result = DeepFace.verify(
                        img1_path=known_image,
                        img2_path=face,
                        model_name="VGG-Face",
                        detector_backend="opencv",
                        enforce_detection=False
                    )

                    if result["verified"]:

                        filename = os.path.basename(
                            known_image
                        )

                        person_name = os.path.splitext(
                            filename
                        )[0]

                        break

                except Exception:
                    pass

            results[i] = person_name

    # ======================================================
    # DRAW FACE BOX
    # ======================================================

    for i, (x, y, w, h) in enumerate(faces):

        name = results.get(i, "Checking...")

        # --------------------------------------------------
        # Rectangle
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # Name
        # --------------------------------------------------

        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ======================================================
    # DISPLAY
    # ======================================================

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "DeepFace Face Recognition",
        frame
    )

    # ======================================================
    # QUIT
    # ======================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================================
# CLOSE
# ==========================================================

camera.release()
cv2.destroyAllWindows()
