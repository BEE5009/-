"""Create training data from collected images.

Input structure:
  data/0/*.jpg  -> A
  data/1/*.jpg  -> B
  ...
  data/25/*.jpg -> Z

Output: data.pickle with keys: data, labels, label_map
"""
import os
import pickle
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit('Please install mediapipe: pip install mediapipe')

DATA_DIR = './data'
OUTPUT_PICKLE = 'data.pickle'

if not os.path.exists(DATA_DIR):
    raise SystemExit(f'Data directory not found: {DATA_DIR}')

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

all_vectors = []
all_labels = []

for class_name in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    for file_name in sorted(os.listdir(class_dir)):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        file_path = os.path.join(class_dir, file_name)
        image = cv2.imread(file_path)
        if image is None:
            print(f'Warning: cannot read image {file_path}')
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            print(f'Warning: no hand landmarks in {file_path}')
            continue

        landmarks = result.multi_hand_landmarks[0].landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]

        if len(x_coords) != 21 or len(y_coords) != 21:
            print(f'Warning: unexpected landmarks count in {file_path}')
            continue

        min_x, min_y = min(x_coords), min(y_coords)
        vector = []
        for lm in landmarks:
            vector.append(lm.x - min_x)
            vector.append(lm.y - min_y)

        all_vectors.append(vector)
        all_labels.append(class_name)

hands.close()

if len(all_vectors) == 0:
    raise SystemExit('No valid data extracted. Check your images and hand detection settings.')

unique_labels = sorted(set(all_labels), key=lambda x: int(x) if x.isdigit() else x)
label_map = {int(lbl): chr(ord('A') + int(lbl)) if lbl.isdigit() else lbl for lbl in unique_labels}

print(f'Total samples: {len(all_vectors)}')
print(f'Unique labels: {unique_labels}')

pickle.dump({
    'data': np.asarray(all_vectors, dtype=np.float32),
    'labels': np.asarray(all_labels),
    'label_map': label_map,
}, open(OUTPUT_PICKLE, 'wb'))

print(f'Wrote {OUTPUT_PICKLE}')
