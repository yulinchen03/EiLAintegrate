import csv
import os
import cv2
from pathlib import Path

current_dir = Path.cwd()
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent

annotations_csv = os.path.join(grandparent_dir, 'Annotations/annotations.csv')
videos_folder = os.path.join(grandparent_dir, 'Videos')
#
outputs_path = current_dir

if not os.path.exists(outputs_path):
    os.makedirs(outputs_path)

with open(annotations_csv, newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)
    next(reader)
    # Loop through each row in the CSV file
    for idx, row in enumerate(reader):
        print(idx, row)
        # Assuming each row has three values, you can save them into variables
        # Change the variable names and number according to your CSV structure
        video_tag, clip_id, label, frame_no, x, y, w, h, person_id = row
        video_folder_path = os.path.join(videos_folder, video_tag)
        video_path = os.path.join(video_folder_path, 'Complete.mp4')

        annotations = 3 - label.count('No annotation')

        if annotations <= 1:
            output_path = os.path.join(outputs_path, 'train')
        else:
            output_path = os.path.join(outputs_path, 'test')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

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

        x = int((float(x)/100) * width)
        y = int((float(y)/100) * height)
        w = int((float(w)/100) * width)
        h = int((float(h)/100) * height)

        text_x = 0
        text_y = h - 20

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Write the string close to the bounding box
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

        cv2.imwrite(output_path + "\\" + str(idx) + '.jpg', frame)