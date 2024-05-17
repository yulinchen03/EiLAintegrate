import os
from PIL import Image
from pathlib import Path

current_dir = Path.cwd()
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent

src_train_directory = os.path.join(parent_dir, '3.Cropped_faces/train')
src_test_directory = os.path.join(parent_dir, '3.Cropped_faces/test')
output_training_set_dir = os.path.join(os.pardir, '4.Resized_image/train')
output_test_set_dir = os.path.join(os.pardir, '4.Resized_image/test')

if not os.path.exists(output_training_set_dir):
    os.makedirs(output_training_set_dir)

if not os.path.exists(output_test_set_dir):
    os.makedirs(output_test_set_dir)

print(src_train_directory)
print(src_test_directory)

for file in os.listdir(src_train_directory):
    img = Image.open(os.path.join(src_train_directory, file))
    img = img.resize((48, 48))
    recolored_img = img.convert('L')
    recolored_img.save(os.path.join(output_training_set_dir, file), 'png')

for file in os.listdir(src_test_directory):
    img = Image.open(os.path.join(src_test_directory, file))
    img = img.resize((48, 48))
    recolored_img = img.convert('L')
    recolored_img.save(os.path.join(output_test_set_dir, file), 'png')
