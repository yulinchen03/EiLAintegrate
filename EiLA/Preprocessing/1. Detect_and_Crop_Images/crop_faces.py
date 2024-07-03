import csv
import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import ast
import shutil


# Decide out of three available annotations, which one to use as the truth label
def check_annotations(annotations):
    if len(annotations) == 3: # if all three annotators annotated the images (No Annotation does not count)
        a, b, c = annotations
        if a == b == c: # if all three annotated the same emotion, return the emotion
            return a
        elif a == b or b == c or a == c: # if two annotated the same emotion and the other one annotated something different, return the dominant emotion
            if a == b:
                return a
            elif b == c:
                return b
            else:
                return c
        else: # if all three annotated differently, return the emotion annotated by the first annotator
            return a
    elif len(annotations) == 2: # if 2 out of 3 annotators annotated the images, return the emotion annotated by the first annotator
        a, b = annotations
        return a
    else:
        return annotations[0] # if only one annotator annoted the images, return the emotion annotated


def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None


current_dir = Path.cwd()
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent

annotations_csv = os.path.join(grandparent_dir, 'Annotations/annotations.csv')
videos_folder = os.path.join(grandparent_dir, 'Videos')

cropped_image_output = os.path.join(parent_dir, '1. Detect_and_Crop_Images') # target output directory

if not os.path.exists(cropped_image_output):
    os.makedirs(cropped_image_output)

model = YOLO("../../../yolov8n-face.pt") # load YOLOv8 model

emotion_labels_EiLA = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
emotion_labels_common = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}

for emotion in emotion_labels_common.keys(): # for the train, validation & test set, create a subdirectory for each emotion
    if not os.path.exists(os.path.join(cropped_image_output, 'EiLA_test', emotion)):
        os.makedirs(os.path.join(cropped_image_output, 'EiLA_test', emotion))
    if not os.path.exists(os.path.join(cropped_image_output, 'EiLA_train', emotion)):
        os.makedirs(os.path.join(cropped_image_output, 'EiLA_train', emotion))
    if not os.path.exists(os.path.join(cropped_image_output, 'EiLA_validation', emotion)):
        os.makedirs(os.path.join(cropped_image_output, 'EiLA_validation', emotion))

# for file in os.listdir(os.path.join(cropped_image_output, 'EiLA_train')):
#     if os.path.isfile(os.path.join(cropped_image_output, 'EiLA_train', file)):
#         emotion_index = int(file.split('_')[2])
#         emotion = get_key_from_value(emotion_labels, emotion_index)
#         shutil.copy(os.path.join(cropped_image_output, 'EiLA_train', file), os.path.join(cropped_image_output, 'EiLA_train', emotion, file))
# for file in os.listdir(os.path.join(cropped_image_output, 'EiLA_test')):
#     if os.path.isfile(file):
#         emotion_index = int(file.split('_')[2])
#         emotion = get_key_from_value(emotion_labels, emotion_index)
#         shutil.copy(os.path.join(cropped_image_output, 'EiLA_test', file), os.path.join(cropped_image_output, 'EiLA_test', emotion, file))
# for file in os.listdir(os.path.join(cropped_image_output, 'EiLA_validation')):
#     if os.path.isfile(os.path.join(cropped_image_output, 'EiLA_validation', file)):
#         emotion_index = int(file.split('_')[2])
#         emotion = get_key_from_value(emotion_labels, emotion_index)
#         shutil.copy(os.path.join(cropped_image_output, 'EiLA_validation', file), os.path.join(cropped_image_output, 'EiLA_validation', emotion, file))

# Open annotations.csv
with open(annotations_csv, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for idx, row in enumerate(reader):
        print(idx, row)
        # Assuming each row has three values, you can save them into variables
        # Change the variable names and number according to your CSV structure
        video_tag, clip_id, label, frame_no, x, y, w, h, person_id = row
        video_folder_path = os.path.join(videos_folder, video_tag)
        video_path = os.path.join(video_folder_path, 'Complete.mp4')

        labels = ast.literal_eval(label)
        annotations = [annotation for annotation in labels if isinstance(annotation, list)]

        final_label = emotion_labels_EiLA[check_annotations(annotations)[0]] # get emotion label in int form (0-6)
        final_emotion = [key for key, value in emotion_labels_common.items() if value == final_label][0]  # get emotion label in string form

        # Assign sample based on number of annotations
        match len(annotations):
            case 1:
                output_folder = 'EiLA_train'
            case 2:
                output_folder = 'EiLA_validation'
            case 3:
                output_folder = 'EiLA_test'

        cropped_image_video_output = os.path.join(cropped_image_output, output_folder)
        if not os.path.exists(cropped_image_video_output):
            os.makedirs(cropped_image_video_output)

        # Draw box around subject in frame using annotations
        cam = cv2.VideoCapture(video_path)
        if not cam.isOpened():
            print("Error: Couldn't open the video file.")
            exit()

        cam.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))

        ret, frame = cam.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Couldn't read the frame.")
            exit()

        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

        x = int((float(x) / 100) * width)
        y = int((float(y) / 100) * height)
        w = int((float(w) / 100) * width)
        h = int((float(h) / 100) * height)

        text_x = 0
        text_y = h - 20

        # Crop and keep the bounding box
        cropped_frame = frame[y:y+h, x:x+w]

        # Use YOLOv8 to detect face in bounding box
        results = model.predict(source=cropped_frame)
        probabilities = results[0].boxes.conf.float().cpu().tolist()
        print(probabilities)
        # no probabilities if no face detected
        if len(probabilities) == 0:
            print("No face detected")
        else:
            boxes = results[0].boxes
            for i in range(len(probabilities)):
                if probabilities[i] < 0.6: # discard samples with <60% confidence that a face is present in frame
                    continue
                else:
                    print("Face detected with probability: " + str(round(probabilities[i] * 100, 2)) + "%")

                    top_left_x = int(boxes[i].xyxy.tolist()[0][0])
                    top_left_y = int(boxes[i].xyxy.tolist()[0][1])
                    bottom_right_x = int(boxes[i].xyxy.tolist()[0][2])
                    bottom_right_y = int(boxes[i].xyxy.tolist()[0][3])

                    cropped_face = cropped_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] # keep only the face from the bounding box

                    # save demographic info in each sample's filename for extraction later on
                    filename = cropped_image_video_output + "\\" + final_emotion + "\\" + str(idx) + '_label_' + str(final_label) + '_video_' + video_tag + '_clip_' + str(clip_id) + '_person_' + str(person_id) + '.jpg'
                    print('writing to file: ' + filename)
                    cv2.imwrite(filename, cropped_face)
