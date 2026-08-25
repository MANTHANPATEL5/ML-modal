from PIL import Image
import os

dataset_path = "dataset"

for class_name in ["cats", "dogs"]:

    folder = os.path.join(dataset_path, class_name)

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)

        try:
            img = Image.open(file_path)

            # Convert everything to RGB
            img = img.convert("RGB")

            # Resize
            img = img.resize((128, 128))

            # Save as JPG
            new_filename = os.path.splitext(filename)[0] + ".jpg"
            new_path = os.path.join(folder, new_filename)

            img.save(new_path, "JPEG")

            # Remove old file if extension changed
            if file_path != new_path:
                os.remove(file_path)

            print("Fixed:", filename)

        except Exception as e:
            print("ERROR:", filename, e)

print("All images processed successfully!")