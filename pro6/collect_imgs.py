"""Collect training data for sign language detection"""
import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define English letters (A-Z = 26 classes)
NUMBER_OF_CLASSES = 26
DATASET_SIZE = 100

cap = cv2.VideoCapture(0)

for j in range(NUMBER_OF_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    letter = chr(ord('A') + j)
    print(f'Collecting data for class {letter}')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            break
        cv2.putText(frame, f'Ready to collect {letter}? Press "Q"',
                   (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect samples
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            break
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()