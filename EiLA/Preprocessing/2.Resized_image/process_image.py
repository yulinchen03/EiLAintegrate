import os
from PIL import Image
from pathlib import Path

current_dir = Path.cwd()
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent

src_train_directory = os.path.join(parent_dir, '1. Detect_and_Crop_Images/EiLA_train')
src_test_directory = os.path.join(parent_dir, '1. Detect_and_Crop_Images/EiLA_test')
src_validation_directory = os.path.join(parent_dir, '1. Detect_and_Crop_Images/EiLA_validation')
output_training_set_dir = os.path.join(os.pardir, '2.Resized_image/EiLA_train')
output_test_set_dir = os.path.join(os.pardir, '2.Resized_image/EiLA_test')
output_validation_set_dir = os.path.join(os.pardir, '2.Resized_image/EiLA_validation')
emotion_labels = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}


if not os.path.exists(output_training_set_dir):
    os.makedirs(output_training_set_dir)
if not os.path.exists(output_test_set_dir):
    os.makedirs(output_test_set_dir)
if not os.path.exists(output_validation_set_dir):
    os.makedirs(output_validation_set_dir)

for emotion in emotion_labels.keys():
    if not os.path.exists(os.path.join(output_test_set_dir, emotion)):
        os.makedirs(os.path.join(output_test_set_dir, emotion))
    if not os.path.exists(os.path.join(output_training_set_dir, emotion)):
        os.makedirs(os.path.join(output_training_set_dir, emotion))
    if not os.path.exists(os.path.join(output_validation_set_dir, emotion)):
        os.makedirs(os.path.join(output_validation_set_dir, emotion))

for dir in os.listdir(src_train_directory):
    for file in os.listdir(os.path.join(src_train_directory, dir)):
        img = Image.open(os.path.join(src_train_directory, dir, file))
        img = img.resize((48, 48)) # resize image to 48x48 pixels
        recolored_img = img.convert('L') # convert image to grayscale
        recolored_img.save(os.path.join(output_training_set_dir, dir, file), 'png')

for dir in os.listdir(src_test_directory):
    for file in os.listdir(os.path.join(src_test_directory, dir)):
        img = Image.open(os.path.join(src_test_directory, dir, file))
        img = img.resize((48, 48))
        recolored_img = img.convert('L')
        recolored_img.save(os.path.join(output_test_set_dir, dir, file), 'png')

for dir in os.listdir(src_validation_directory):
    for file in os.listdir(os.path.join(src_validation_directory, dir)):
        img = Image.open(os.path.join(src_validation_directory, dir, file))
        img = img.resize((48, 48))
        recolored_img = img.convert('L')
        recolored_img.save(os.path.join(output_validation_set_dir, dir, file), 'png')
