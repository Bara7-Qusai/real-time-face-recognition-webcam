import os
import cv2
import face_recognition
import csv
from datetime import datetime

# Path to the directory containing known people's images
known_faces_dir = 'path_to_known_faces_directory'

# Load known people's images and learn their face encodings
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Check for faces in the image and extract encodings
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Open a CSV file to write names with date and time
with open('recognized_faces.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Date", "Time"])

    # Start the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = frame[:, :, ::-1]
        
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # If a match is found, use the known person's name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Write the person's name above the rectangle
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save the person's name with date and time in the CSV file
            now = datetime.now()
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
        
        # Display the frame with names
        cv2.imshow('Face Detection', frame)
        
        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
