import csv
import os
import cv2
from ultralytics import YOLO
from pathlib import Path

current_dir = Path.cwd()
parent_dir = current_dir.parent
grandparent_dir = parent_dir.parent

annotations_csv = os.path.join(grandparent_dir, 'Annotations/annotations.csv')
videos_folder = os.path.join(grandparent_dir, 'Videos')

face_detection_output = os.path.join(parent_dir, '2.Face_detection')

if not os.path.exists(face_detection_output):
    os.makedirs(face_detection_output)

model = YOLO("../../../yolov8n-face.pt")

# Open the CSV file
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
            output_folder = 'train'
        else:
            output_folder = 'test'

        face_detection_video_output = os.path.join(face_detection_output, output_folder)
        if not os.path.exists(face_detection_video_output):
            os.makedirs(face_detection_video_output)

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

        cropped_frame = frame[y:y+h, x:x+w]

        results = model.predict(source=cropped_frame)
        probabilities = results[0].boxes.conf.float().cpu().tolist()
        print(probabilities)
        if len(probabilities) == 0:
            print("No face detected")
        else:
            boxes = results[0].boxes
            for i in range(len(probabilities)):
                if probabilities[i] < 0.6:
                    continue
                else:
                    print("Face detected with probability: " + str(round(probabilities[i] * 100, 2)) + "%")

                    top_left_x = int(boxes[i].xyxy.tolist()[0][0])
                    top_left_y = int(boxes[i].xyxy.tolist()[0][1])
                    bottom_right_x = int(boxes[i].xyxy.tolist()[0][2])
                    bottom_right_y = int(boxes[i].xyxy.tolist()[0][3])

                    cv2.rectangle(cropped_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
                    cv2.putText(cropped_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

                    cv2.imwrite(face_detection_video_output + "\\" + str(idx) + '.jpg', cropped_frame)


