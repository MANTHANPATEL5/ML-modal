import cv2
import os

# ==========================================================
# SETTINGS
# ==========================================================

SAVE_FOLDER = "known_faces"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# Ask user for name
NAME = input("Enter person's name: ").strip()

if not NAME:
    print("❌ Name cannot be empty.")
    exit()

# ==========================================================
# LOAD FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================================================
# OPEN CAMERA
# ==========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera not found")
    exit()

print("Camera started.")
print("Put your face inside the box.")
print("Press SPACE to save.")
print("Press Q to quit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("❌ Could not read camera")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # ======================================================
    # DRAW FACE BOX
    # ======================================================

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ======================================================
    # DISPLAY INSTRUCTIONS
    # ======================================================

    cv2.putText(
        frame,
        f"Name: {NAME} | SPACE = Save | Q = Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    # ======================================================
    # SPACE = SAVE FACE
    # ======================================================

    if key == 32:

        if len(faces) == 0:

            print("❌ No face detected. Try again.")

        elif len(faces) > 1:

            print("❌ Multiple faces detected.")
            print("Please keep only one person in the camera.")

        else:

            # Get face coordinates
            x, y, w, h = faces[0]

            # Crop only the face
            face = frame[y:y+h, x:x+w]

            file_path = os.path.join(
                SAVE_FOLDER,
                NAME + ".jpg"
            )

            cv2.imwrite(file_path, face)

            print("✅ Face detected and saved:")
            print(file_path)

            break

    # ======================================================
    # Q = QUIT
    # ======================================================

    elif key == ord("q"):
        print("Registration cancelled.")
        break

camera.release()
cv2.destroyAllWindows()
