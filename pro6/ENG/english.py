"""
English Sign Language Detector
Detects hand signs and recognizes English letters A-Z using MediaPipe and OpenCV
"""

import pickle
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_SELECTED_THAI_FONT_PATH: Optional[str] = None

# Optional: Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mediapipe tasks model URL (hand landmarker task file)
MEDIA_PIPE_HAND_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'

# Try to import MediaPipe Tasks (v0.10+)
try:
    from mediapipe.tasks.python.vision import hand_landmarker as hl_module
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as vrm
    MP_TASKS_AVAILABLE = True
except Exception:
    MP_TASKS_AVAILABLE = False


def _draw_unicode_text(img, text, position, font_size=32, color=(0, 255, 0)):
    """Draw Unicode text (e.g., Thai) onto an OpenCV image.
    Falls back to cv2.putText when Pillow is not available.
    """
    if not _PIL_AVAILABLE:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return

    from numpy import array

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    def _find_thai_font_path():
        """Find a font file that supports Thai text (prefers Sarabun)."""
        filename_candidates = [
            "Sarabun-Regular.ttf", "TH Sarabun New.ttf", "THSarabun.ttf",
            "THSarabunNew.ttf", "Leelawadee UI.ttf", "NotoSansThai-Regular.ttf",
        ]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_font_dirs = [script_dir, os.getcwd(), os.path.join(script_dir, "ฟอนต์")]

        for base in local_font_dirs:
            for name in filename_candidates:
                path = os.path.join(base, name)
                if os.path.exists(path):
                    return path
            if os.path.isdir(base):
                for root, _, files in os.walk(base):
                    for file in files:
                        if file.lower().endswith(".ttf") and "sarabun" in file.lower():
                            return os.path.join(root, file)

        system_font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts") if os.name == "nt" else "/usr/share/fonts"
        if os.path.isdir(system_font_dir):
            for root, _, files in os.walk(system_font_dir):
                for file in files:
                    if file.lower().endswith(".ttf") and "sarabun" in file.lower():
                        return os.path.join(root, file)
        return None

    global _SELECTED_THAI_FONT_PATH
    if _SELECTED_THAI_FONT_PATH is None:
        _SELECTED_THAI_FONT_PATH = _find_thai_font_path()

    if _SELECTED_THAI_FONT_PATH:
        try:
            font = ImageFont.truetype(_SELECTED_THAI_FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    img[:] = cv2.cvtColor(array(img_pil), cv2.COLOR_RGB2BGR)


# Standard ASL (American Sign Language) Fingerspelling A-Z Descriptions (Thai)
ASL_DESCRIPTIONS = {
    'A': 'โป้งชี้ออกข้าง นิ้วอื่นหุบเป็นหมัด',
    'B': 'ฝ่ามือเปิด นิ้วทั้งหมดชี้ขึ้น โป้งพับเข้า',
    'C': 'ปัดนิ้ว เป็นรูป C โค้ง',
    'D': 'นิ้วชี้ชี้ขึ้น นิ้วอื่นหุบ โป้งไปข้าง',
    'E': 'นิ้วทั้งหมดโค้งเข้า เป็นท่ากำ',
    'F': 'นิ้วชี้และกลาง ชี้ขึ้น โป้งอยู่ระหว่าง',
    'G': 'นิ้วชี้และโป้งชี้ไปข้าง นิ้วอื่นหุบ',
    'H': 'นิ้วชี้และกลาง ชี้ขึ้นเคียงข้าง',
    'I': 'นิ้วก้อย ชี้ขึ้น โป้งก็ชี้ข้างเคียง',
    'J': 'นิ้วก้อย ชี้ขึ้นเป็นรูป J',
    'K': 'นิ้วกลาง ชี้ขึ้น นิ้วชี้และโป้ง เป็นรูป K',
    'L': 'นิ้วชี้และโป้ง ทำมุมฉาก เป็นรูป L',
    'M': 'นิ้วชี้ กลาง และนาง ชี้ขึ้น โป้งพับเข้า',
    'N': 'นิ้วกลางและนาง ชี้ขึ้น โป้งพับเข้า',
    'O': 'นิ้วทั้งหมดโค้ง เป็นรูป O',
    'P': 'นิ้วชี้และกลาง ชี้ลง โป้งชี้ข้าง',
    'Q': 'นิ้วชี้และโป้ง ชี้ลง เป็นรูป Q',
    'R': 'นิ้วชี้และกลาง ขวางทับกัน',
    'S': 'หมัดมือ โป้งด้านนอก',
    'T': 'โป้งอยู่ระหว่าง นิ้วชี้และกลาง',
    'U': 'นิ้วชี้และกลาง ชี้ขึ้นเคียง เป็นรูป U',
    'V': 'นิ้วชี้และกลาง ชี้ขึ้นแยก เป็นรูป V',
    'W': 'นิ้วชี้ กลาง นาง ชี้ขึ้น เป็นรูป W',
    'X': 'นิ้วชี้ โค้งคดเป็นรูป X',
    'Y': 'โป้งและนิ้วก้อย ชี้ขึ้น นิ้วอื่นหุบ',
    'Z': 'นิ้วชี้ วาด Z ในอากาศ'
}


class EnglishSignLanguageDetector:
    """
    Real-time sign language detector for English letters using MediaPipe Hands
    and a trained RandomForest classifier.
    """

    def __init__(self, model_path: str = 'model.p', camera_index: int = 0,
                 min_detection_confidence: float = 0.5):
        """
        Initialize the detector with a trained model.

        Args:
            model_path: Path to the trained model pickle file
            camera_index: Webcam index (0 for default, 1, 2 for other cameras)
            min_detection_confidence: Minimum confidence threshold for hand detection
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.min_detection_confidence = min_detection_confidence

        # Initialize MediaPipe hand detector: prefer Tasks API (0.10+) else fallback to solutions (if present)
        self.hand_landmarker = None
        self.mp_hands = None
        self.mp_drawing = None
        self.mp_drawing_styles = None

        if MP_TASKS_AVAILABLE:
            self._init_task_hand_landmarker(max_num_hands=2,
                                            min_detection_confidence=min_detection_confidence,
                                            min_tracking_confidence=0.5)
        else:
            if hasattr(mp, 'solutions'):
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5
                )
            else:
                print("Error: mediapipe does not support solutions or tasks in this environment.")
                print("Install mediapipe>=0.10 and ensure tasks API is available.")

        # Load trained model
        self.model = None
        self.labels_dict = {}
        self.load_model()

        # Webcam capture
        self.cap = None
        self.is_running = False
        
        # Recording system (for 'r', 'e', 't' keys)
        self.recorded_letters = []
        self.recognized_words = []

    def load_model(self) -> bool:
        """Load the trained RandomForest model."""
        if not os.path.exists(self.model_path):
            print(f"Error: Model file '{self.model_path}' not found!")
            print("Please train the model first using create_dataset.py and train_classifier.py")
            return False

        try:
            model_dict = pickle.load(open(self.model_path, 'rb'))
            self.model = model_dict.get('model')
            self.labels_dict = model_dict.get('labels_dict', {})

            if self.model is None:
                print("Error: Invalid model file format")
                return False

            print(f"Model loaded successfully from {self.model_path}")
            print(f"Recognized classes: {len(self.labels_dict)} - {sorted(self.labels_dict.values())}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_asl_description(self, letter: str) -> str:
        """
        Get standard ASL fingerspelling description for a letter.
        
        Args:
            letter: Single uppercase letter (A-Z)
            
        Returns:
            Description of how the letter is signed in ASL
        """
        return ASL_DESCRIPTIONS.get(letter.upper(), "Unknown letter")

    def _download_task_model(self, target_path: str):
        """Download the MediaPipe hand_landmarker task model if missing."""
        try:
            print(f"Downloading model to {target_path}...")
            urllib.request.urlretrieve(MEDIA_PIPE_HAND_LANDMARKER_MODEL_URL, target_path)
            print("Download completed")
        except Exception as exc:
            print(f"Failed to download model: {exc}")
            raise

    def _init_task_hand_landmarker(self, max_num_hands: int, min_detection_confidence: float, min_tracking_confidence: float):
        """Initialize MediaPipe Hands Landmarker through Tasks API."""
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            self._download_task_model(model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vrm.VisionTaskRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.hand_landmarker = HandLandmarker.create_from_options(options)

    def extract_hand_landmarks(self, frame: np.ndarray) -> tuple:
        """
        Extract hand landmarks from a frame using MediaPipe.

        Returns:
            Tuple of (all_data_vectors, hand_positions, processed_frame)
            where hand_positions contains (x1, y1, x2, y2) for bounding boxes
        """
        H, W, _ = frame.shape

        all_data_vectors = []
        hand_positions = []

        if self.hand_landmarker is not None:
            image = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(time.time() * 1000)
            result = self.hand_landmarker.detect_for_video(image, timestamp_ms)
            landmarks_list = getattr(result, 'hand_landmarks', None)

            if landmarks_list:
                for hand_landmarks in landmarks_list:
                    # support both normalized landmark list and native iterable
                    if hasattr(hand_landmarks, 'landmark'):
                        points = list(hand_landmarks.landmark)
                    else:
                        points = list(hand_landmarks)

                    x_coords = [pt.x for pt in points]
                    y_coords = [pt.y for pt in points]

                    if not x_coords or not y_coords:
                        continue

                    min_x = min(x_coords)
                    min_y = min(y_coords)

                    data_aux = []
                    for pt in points:
                        data_aux.append(pt.x - min_x)
                        data_aux.append(pt.y - min_y)

                    all_data_vectors.append(data_aux)

                    x1 = int(min_x * W) - 10
                    y1 = int(min_y * H) - 10
                    x2 = int(max(x_coords) * W) + 10
                    y2 = int(max(y_coords) * H) + 10

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)

                    hand_positions.append((x1, y1, x2, y2))

        elif hasattr(mp, 'solutions') and hasattr(self, 'hands') and self.hands is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.mp_drawing is not None:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                for hand_landmarks in results.multi_hand_landmarks:
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]

                    if not x_coords or not y_coords:
                        continue

                    min_x = min(x_coords)
                    min_y = min(y_coords)

                    data_aux = []
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min_x)
                        data_aux.append(lm.y - min_y)

                    all_data_vectors.append(data_aux)

                    x1 = int(min_x * W) - 10
                    y1 = int(min_y * H) - 10
                    x2 = int(max(x_coords) * W) + 10
                    y2 = int(max(y_coords) * H) + 10

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)
                    hand_positions.append((x1, y1, x2, y2))

        return all_data_vectors, hand_positions, frame

    def predict_signs(self, frame: np.ndarray) -> tuple:
        """
        Predict sign language in a frame.

        Returns:
            Tuple of (frame_with_predictions, predictions)
            where predictions is list of (character, confidence, bbox)
        """
        data_vectors, hand_positions, frame = self.extract_hand_landmarks(frame)
        predictions = []

        if data_vectors and self.model:
            for i, data_aux in enumerate(data_vectors):
                try:
                    prediction = self.model.predict([np.asarray(data_aux)])[0]
                    confidence = self.model.predict_proba([np.asarray(data_aux)]).max()

                    predicted_character = self.labels_dict.get(int(prediction), '?')

                    if i < len(hand_positions):
                        x1, y1, x2, y2 = hand_positions[i]
                        predictions.append((predicted_character, confidence, (x1, y1, x2, y2)))

                        # Draw bounding box and prediction
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(
                            frame,
                            f'{predicted_character} ({confidence:.2f})',
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA
                        )
                except Exception as e:
                    print(f"Prediction error: {e}")

        return frame, predictions

    def run_live_detection(self, show_fps: bool = True, exit_key: int = ord('q')):
        """
        Run live detection from webcam.

        Args:
            show_fps: Display FPS counter
            exit_key: Key code to exit (default: 'q')
            
        Controls:
            'q' or ESC: Exit
            'r': Record current detected letter
            'e': Erase last recorded letter
            't': Finish and display result
            'p': Print all recognized words
        """
        if self.model is None:
            print("Model not loaded. Cannot start detection.")
            return

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return

        print("Starting live detection (press 'q' to exit)...")
        print("Controls: 'r'=Record, 'e'=Erase, 't'=Finish, 'p'=Print")
        self.is_running = True
        frame_count = 0
        fps_start_time = cv2.getTickCount()
        fps = 0.0  # Initialize fps to avoid UnboundLocalError
        current_prediction = None

        try:
            while self.is_running:
                ret, frame = self.cap.read()

                if not ret:
                    print("Error: Failed to read frame")
                    break

                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)

                # Predict signs
                frame, predictions = self.predict_signs(frame)
                
                # Store current prediction for 'r' key
                if predictions:
                    current_prediction = predictions[0][0]

                # Display FPS
                if show_fps:
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = (cv2.getTickCount() - fps_start_time) / cv2.getTickFrequency()
                        if elapsed > 0:
                            fps = frame_count / elapsed
                        fps_start_time = cv2.getTickCount()
                        frame_count = 0

                    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display recorded letters
                recorded_text = ''.join(self.recorded_letters)
                if recorded_text:
                    cv2.putText(frame, f'Recorded: {recorded_text}', (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

                # Display ASL description for current prediction
                if current_prediction and current_prediction != '?':
                    description = self.get_asl_description(current_prediction)
                    _draw_unicode_text(frame, f'Standard ASL: {description}', (10, 110), 
                                     font_size=24, color=(0, 165, 255))

                # Display instructions
                cv2.putText(frame, "Press 'r'=Record, 'e'=Erase, 't'=Finish, 'q'=Quit", (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

                # Show frame
                cv2.imshow('English Sign Language Detector', frame)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == exit_key or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Record current letter
                    if current_prediction and current_prediction != '?':
                        self.recorded_letters.append(current_prediction)
                        print(f"Record: {current_prediction} | Total: {''.join(self.recorded_letters)}")
                    else:
                        print("No valid letter detected to record")
                elif key == ord('e'):  # Erase last letter
                    if self.recorded_letters:
                        removed = self.recorded_letters.pop()
                        print(f"Erased: {removed} | Remaining: {''.join(self.recorded_letters)}")
                    else:
                        print("No letters to erase")
                elif key == ord('t'):  # Finish and display result
                    if self.recorded_letters:
                        self.display_recording_details()
                        self.recorded_letters = []
                    else:
                        print("No letters recorded")
                elif key == ord('p'):  # Print recognized words
                    print('Recognized words:', self.recognized_words)

        finally:
            self.stop_detection()

    def stop_detection(self):
        """Stop live detection and clean up resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

    def display_recording_details(self):
        """Display detailed ASL descriptions for all recorded letters."""
        if not self.recorded_letters:
            return
        
        result_text = ''.join(self.recorded_letters)
        print(f"\n{'='*60}")
        print(f"RECORDED: {result_text}")
        print(f"{'='*60}")
        
        for i, letter in enumerate(self.recorded_letters, 1):
            description = self.get_asl_description(letter)
            print(f"{i}. {letter}: {description}")
        
        print(f"{'='*60}\n")

    def detect_from_image(self, image_path: str) -> tuple:
        """
        Detect signs in a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (frame_with_predictions, predictions)
        """
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None, []

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Cannot read image: {image_path}")
            return None, []

        frame, predictions = self.predict_signs(frame)
        return frame, predictions

    def display_image_detection(self, image_path: str, wait_ms: int = 0):
        """
        Display image with detected signs.

        Args:
            image_path: Path to the image
            wait_ms: Time to display (0 = wait for keypress)
        """
        frame, predictions = self.detect_from_image(image_path)

        if frame is not None:
            cv2.imshow('Sign Detection', frame)
            print(f"Detected signs: {predictions}")
            cv2.waitKey(wait_ms)
            cv2.destroyAllWindows()


def main():
    """
    Main entry point for the English Sign Language Detector.
    """
    # Initialize detector
    detector = EnglishSignLanguageDetector(
        model_path='model.p',  # Pre-trained model path
        camera_index=0,         # Use default camera
        min_detection_confidence=0.5
    )

    # Start live detection from webcam
    detector.run_live_detection()

    # Optional: Detect from image file
    # frame, predictions = detector.detect_from_image('test_image.jpg')
    # if frame is not None:
    #     detector.display_image_detection('test_image.jpg')


if __name__ == '__main__':
    main()
