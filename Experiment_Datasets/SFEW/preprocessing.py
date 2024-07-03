import os
from PIL import Image
from pathlib import Path

current_dir = Path.cwd()
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent

output_training_set_dir = 'Processed_Train_Faces'
output_validation_set_dir = 'Processed_Validation_Faces'

if not os.path.exists(output_training_set_dir):
    os.makedirs(output_training_set_dir)

if not os.path.exists(output_validation_set_dir):
    os.makedirs(output_validation_set_dir)

for dir in os.listdir(current_dir):
    if dir.endswith('.py'):
        continue
    if 'Train' in dir:
        for sub_dir in os.listdir(dir):
            if not os.path.exists(os.path.join(output_training_set_dir, sub_dir)):
                os.makedirs(os.path.join(output_training_set_dir, sub_dir))
            for file in os.listdir(os.path.join(dir, sub_dir)):
                print('Processing image: ' + file)
                img = Image.open(os.path.join(dir, sub_dir, file))
                img = img.resize((48, 48))
                recolored_img = img.convert('L')
                recolored_img.save(os.path.join(output_training_set_dir, sub_dir, file), 'png')
        print('Training set preprocessing complete.')
    elif 'Val' in dir:
        for sub_dir in os.listdir(dir):
            if not os.path.exists(os.path.join(output_validation_set_dir, sub_dir)):
                os.makedirs(os.path.join(output_validation_set_dir, sub_dir))
            for file in os.listdir(os.path.join(dir, sub_dir)):
                print('Processing image: ' + file)
                img = Image.open(os.path.join(dir, sub_dir, file))
                img = img.resize((48, 48))
                recolored_img = img.convert('L')
                recolored_img.save(os.path.join(output_validation_set_dir, sub_dir, file), 'png')
        print('Validation set preprocessing complete.')
    else:
        continue
print('Preprocessing complete.')