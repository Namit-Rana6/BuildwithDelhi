from ultralytics import YOLO
import cv2
from PIL import Image
import os

# Load the trained model
model = YOLO('/content/best (2).pt')

# Path to the test images
test_dir = '/tmp/hackathon_dataset/HackByte_Dataset/data/test/images'

# Get a list of all image files in the test directory
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Perform prediction on each image and display the results
for i, image_path in enumerate(test_images):
    if i < 5:
        results = model(image_path)
        im_array = results[0].plot()
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        display(im)

# Validate the model on the test set
metrics = model.val(data='/tmp/hackathon_dataset/HackByte_Dataset/yolo_params.yaml', split='test')
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
