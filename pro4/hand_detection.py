import argparse
import time
import tempfile
import urllib.request
import os
import sys
from typing import Optional

import cv2

_SELECTED_THAI_FONT_PATH: Optional[str] = None

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


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
            "Sarabun-Regular.ttf",
            "TH Sarabun New.ttf",
            "THSarabun.ttf",
            "THSarabunNew.ttf",
            "THSarabun Bold.ttf",
            "THSarabunNew Bold.ttf",
            "Leelawadee UI.ttf",
            "NotoSansThai-Regular.ttf",
            "Tahoma.ttf",
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

        system_font_dirs = []
        if os.name == "nt":
            system_font_dirs.append(os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"))
        else:
            system_font_dirs.extend(["/usr/share/fonts", "/usr/local/share/fonts", "/Library/Fonts"])

        for base in system_font_dirs:
            for name in filename_candidates:
                path = os.path.join(base, name)
                if os.path.exists(path):
                    return path
            if os.path.isdir(base):
                for root, _, files in os.walk(base):
                    for file in files:
                        if file.lower().endswith(".ttf") and "sarabun" in file.lower():
                            return os.path.join(root, file)

        for name in ["Leelawadee UI.ttf", "Tahoma.ttf", "NotoSansThai-Regular.ttf"]:
            try:
                ImageFont.truetype(name, font_size)
                return name
            except Exception:
                pass

        return None

    global _SELECTED_THAI_FONT_PATH
    if _SELECTED_THAI_FONT_PATH is None:
        _SELECTED_THAI_FONT_PATH = _find_thai_font_path()
        if _SELECTED_THAI_FONT_PATH:
            try:
                print(f"[font] ใช้ฟอนต์ไทย: {_SELECTED_THAI_FONT_PATH}")
            except Exception:
                pass

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


# Thai alphabet ก-ฮ (44 consonants)
THAI_ALPHABET = [
    'ก', 'ข', 'ค', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ',  # 0-9
    'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'พ',  # 10-19
    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส',  # 20-29
    'ห', 'ฮ'                                              # 30-31
]


def classify_gesture(hand_landmarks):
    """Classify hand gesture to Thai letters (ก-ฮ) and basic gestures.
    
    Maps different hand positions to Thai letters based on finger positions.
    """
    fingers_extended = []
    thumb_extended = hand_landmarks[4].x > hand_landmarks[3].x  # Compare x for thumb
    tip_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky
    
    for tip_id in tip_ids:
        pip_id = tip_id - 2
        if hand_landmarks[tip_id].y < hand_landmarks[pip_id].y:
            fingers_extended.append(True)
        else:
            fingers_extended.append(False)
    
    num_extended = sum(fingers_extended)
    
    # Map finger patterns to Thai letters
    # All fingers extended = ก
    if all(fingers_extended) and thumb_extended:
        return THAI_ALPHABET[0]  # ก
    # All fingers extended without thumb = ข
    elif all(fingers_extended) and not thumb_extended:
        return THAI_ALPHABET[1]  # ข
    # Index + middle extended = ค
    elif fingers_extended[0] and fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3]:
        return THAI_ALPHABET[2]  # ค
    # Only index extended = ง
    elif fingers_extended[0] and not any(fingers_extended[1:]):
        return THAI_ALPHABET[3]  # ง
    # Only middle extended = จ
    elif fingers_extended[1] and not any([fingers_extended[0], fingers_extended[2], fingers_extended[3]]):
        return THAI_ALPHABET[4]  # จ
    # Index + middle + ring extended = ฉ
    elif fingers_extended[0] and fingers_extended[1] and fingers_extended[2] and not fingers_extended[3]:
        return THAI_ALPHABET[5]  # ฉ
    # Only ring extended = ช
    elif fingers_extended[2] and not any([fingers_extended[0], fingers_extended[1], fingers_extended[3]]):
        return THAI_ALPHABET[6]  # ช
    # Only pinky extended = ซ
    elif fingers_extended[3] and not any(fingers_extended[:-1]):
        return THAI_ALPHABET[7]  # ซ
    # Index + pinky extended = ฌ
    elif fingers_extended[0] and fingers_extended[3] and not fingers_extended[1] and not fingers_extended[2]:
        return THAI_ALPHABET[8]  # ฌ
    # Thumb extended only = ญ
    elif thumb_extended and not any(fingers_extended):
        return THAI_ALPHABET[9]  # ญ
    # No fingers extended + fist = ด
    elif not any(fingers_extended) and not thumb_extended:
        return THAI_ALPHABET[10]  # ด
    # Two fingers (middle + ring) = ต
    elif not fingers_extended[0] and fingers_extended[1] and fingers_extended[2] and not fingers_extended[3]:
        return THAI_ALPHABET[11]  # ต
    # Three fingers (index, ring, pinky) = ถ
    elif fingers_extended[0] and not fingers_extended[1] and fingers_extended[2] and fingers_extended[3]:
        return THAI_ALPHABET[12]  # ถ
    # Middle + pinky = ท
    elif not fingers_extended[0] and fingers_extended[1] and not fingers_extended[2] and fingers_extended[3]:
        return THAI_ALPHABET[13]  # ท
    
    # Default patterns for remaining letters (32 variations are possible with 5 digits)
    # Calculate a pattern index for remaining letters
    pattern_index = num_extended + (1 if thumb_extended else 0)
    
    # Cycle through remaining alphabet if pattern repeats
    if pattern_index < len(THAI_ALPHABET):
        return THAI_ALPHABET[pattern_index]
    
    return "?"  # Unknown gesture


# Template pose for the word "บ้านใหม่" (used by pose matching).
_BANMAI_TEMPLATE: Optional[list] = None


def _normalize_landmarks(landmarks):
    """Normalize landmarks (relative position + scale) for pose matching."""
    if not landmarks:
        return []

    # Use wrist (landmark 0) as origin.
    origin = landmarks[0]
    coords = [(lm.x - origin.x, lm.y - origin.y, lm.z - origin.z) for lm in landmarks]

    max_dist = max(((x * x + y * y + z * z) ** 0.5 for x, y, z in coords), default=1.0)
    if max_dist <= 0:
        max_dist = 1.0

    return [(x / max_dist, y / max_dist, z / max_dist) for x, y, z in coords]


def _landmark_distance(a, b):
    """Mean Euclidean distance between two normalized landmark sets."""
    if not a or not b or len(a) != len(b):
        return float('inf')

    total = 0.0
    for (ax, ay, az), (bx, by, bz) in zip(a, b):
        dx = ax - bx
        dy = ay - by
        dz = az - bz
        total += (dx * dx + dy * dy + dz * dz) ** 0.5

    return total / len(a)


def is_banmai_pose(hand_landmarks, threshold: float = 0.12):
    """Return True when the current hand pose matches the stored 'บ้านใหม่' template."""
    global _BANMAI_TEMPLATE
    if _BANMAI_TEMPLATE is None:
        return False

    norm = _normalize_landmarks(hand_landmarks)
    dist = _landmark_distance(norm, _BANMAI_TEMPLATE)
    return dist < threshold


def save_banmai_template(hand_landmarks):
    """Store the current hand pose as the 'บ้านใหม่' template."""
    global _BANMAI_TEMPLATE
    _BANMAI_TEMPLATE = _normalize_landmarks(hand_landmarks)
    return _BANMAI_TEMPLATE is not None


def clear_banmai_template():
    """Clear any stored 'บ้านใหม่' pose template."""
    global _BANMAI_TEMPLATE
    _BANMAI_TEMPLATE = None


DEFAULT_TASK_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'


def download_model(url: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.task')
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise


def open_capture(camera_index: int = 0, video_path: Optional[str] = None):
    """Open a video capture from a webcam or a video file.

    On Windows, using CAP_DSHOW often improves camera access.
    """

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video file: {video_path}")
        return cap

    for idx in range(camera_index, camera_index + 3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Using camera index {idx}")
            return cap
        cap.release()
    print(f"Cannot open any camera (tried indexes {camera_index}-{camera_index+2})")
    return cv2.VideoCapture(camera_index)


def run_with_solutions(cap, max_num_hands: int, min_detection_confidence: float):
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    ) as hands:
        prev_time = 0
        recognized_words = []
        recorded_letters = []  # Store recorded letters for 'r' and 't' feature
        current_word = None
        last_landmarks = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Exiting.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_word = None
            last_landmarks = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                    )
                    last_landmarks = hand_landmarks.landmark
                    gesture = classify_gesture(last_landmarks)
                    if is_banmai_pose(last_landmarks):
                        current_word = "บ้านใหม่"
                    else:
                        current_word = gesture

            if current_word:
                if not recognized_words or recognized_words[-1] != current_word:
                    recognized_words.append(current_word)

            # Display current gesture detected
            if current_word:
                _draw_unicode_text(image, current_word, (10, 60), font_size=30, color=(0, 255, 0))

            # Display recorded letters (from 'r' key)
            recorded_text = ''.join(recorded_letters)
            if recorded_text:
                _draw_unicode_text(image, f"บันทึก: {recorded_text}", (10, 120), font_size=26, color=(255, 0, 0))

            # Show whether the 'บ้านใหม่' pose template is set
            template_status = "เซฟแล้ว" if _BANMAI_TEMPLATE else "ยังไม่เซฟ"
            _draw_unicode_text(image, f"ท่า บ้านใหม่: {template_status}", (10, 160), font_size=22, color=(255, 255, 0))

            #FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Hand Detection (press q to quit)', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b'):  # Save current pose as "บ้านใหม่"
                if last_landmarks:
                    if save_banmai_template(last_landmarks):
                        print("บันทึกท่า 'บ้านใหม่' เรียบร้อยแล้ว")
                    else:
                        print("ไม่สามารถบันทึกท่า 'บ้านใหม่' ได้")
                else:
                    print("ยังไม่พบมือ (วางมือไว้ในกล้องก่อนบันทึก)")
            elif key == ord('c'):  # Clear saved template
                clear_banmai_template()
                print("ล้างท่า 'บ้านใหม่' เรียบร้อยแล้ว")
            elif key == ord('r'):  # Record current gesture
                if current_word and current_word != '?':
                    recorded_letters.append(current_word)
                    print(f"บันทึก: {current_word} | รวม: {''.join(recorded_letters)}")
            elif key == ord('e'):  # Remove last recorded letter
                if recorded_letters:
                    removed = recorded_letters.pop()
                    print(f"ลบ: {removed} | เหลือ: {''.join(recorded_letters)}")
                else:
                    print("ไม่มีตัวอักษรให้ลบ")
            elif key == ord('t'):  # Finish and output
                if recorded_letters:
                    result_text = ''.join(recorded_letters)
                    print(f"\n✓ ผลลัพธ์: {result_text}\n")
                    recorded_letters = []
            elif key == ord('p'):
                print('Recognized words:', recognized_words)


def run_with_tasks(cap, model_path: str, max_num_hands: int, min_detection_confidence: float):
    import mediapipe as mp
    # import task modules
    from mediapipe.tasks.python.vision import hand_landmarker as hl_module
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as vrm

    base_options = BaseOptions(model_asset_path=model_path)
    running_mode = getattr(vrm.VisionTaskRunningMode, 'VIDEO', vrm.VisionTaskRunningMode.IMAGE)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)

    prev_time = 0
    recognized_words = []
    recorded_letters = []  # Store recorded letters for 'r' and 't' feature
    current_word = None
    last_landmarks = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Exiting.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            image_out = frame.copy()
            h, w, _ = image_out.shape

            current_word = None
            last_landmarks = None
            if result and getattr(result, 'hand_landmarks', None):
                for hand_landmarks in result.hand_landmarks:
                    pts = []
                    for lm in hand_landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        pts.append((x, y))
                        cv2.circle(image_out, (x, y), 3, (0, 255, 0), -1)

                    last_landmarks = hand_landmarks
                    gesture = classify_gesture(hand_landmarks)
                    if is_banmai_pose(hand_landmarks):
                        current_word = "บ้านใหม่"
                    else:
                        current_word = gesture

                    try:
                        connections = hl_module.HandLandmarksConnections.HAND_CONNECTIONS
                        for conn in connections:
                            start = (int(hand_landmarks[conn.start].x * w), int(hand_landmarks[conn.start].y * h))
                            end = (int(hand_landmarks[conn.end].x * w), int(hand_landmarks[conn.end].y * h))
                            cv2.line(image_out, start, end, (0, 255, 255), 2)
                    except Exception:
                        pass

            if current_word:
                if not recognized_words or recognized_words[-1] != current_word:
                    recognized_words.append(current_word)

            # Display current gesture detected
            if current_word:
                _draw_unicode_text(image_out, current_word, (10, 60), font_size=30, color=(0, 255, 0))

            # Display recorded letters (from 'r' key)
            recorded_text = ''.join(recorded_letters)
            if recorded_text:
                _draw_unicode_text(image_out, f"บันทึก: {recorded_text}", (10, 120), font_size=26, color=(255, 0, 0))

            # Show whether the 'ลับ' pose template is set
            template_status = "เซฟแล้ว" if _BANMAI_TEMPLATE else "ยังไม่เซฟ"
            _draw_unicode_text(image_out, f"ท่า ลับ: {template_status}", (10, 160), font_size=22, color=(255, 255, 0))

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(image_out, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Hand Detection (press q to quit)', image_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b'):  # Save current pose as "ลับ"
                if last_landmarks:
                    if save_banmai_template(last_landmarks):
                        print("บันทึกท่า 'ลับ' เรียบร้อยแล้ว")
                    else:
                        print("ไม่สามารถบันทึกท่า 'ลับ' ได้")
                else:
                    print("ยังไม่พบมือ (วางมือไว้ในกล้องก่อนบันทึก)")
            elif key == ord('c'):  # Clear saved template
                clear_banmai_template()
                print("ล้างท่า 'ลับ' เรียบร้อยแล้ว")
            elif key == ord('r'):  # Record current gesture
                if current_word and current_word != '?':
                    recorded_letters.append(current_word)
                    print(f"บันทึก: {current_word} | รวม: {''.join(recorded_letters)}")
            elif key == ord('e'):  # Remove last recorded letter
                if recorded_letters:
                    removed = recorded_letters.pop()
                    print(f"ลบ: {removed} | เหลือ: {''.join(recorded_letters)}")
                else:
                    print("ไม่มีตัวอักษรให้ลบ")
            elif key == ord('t'):  # Finish and output
                if recorded_letters:
                    result_text = ''.join(recorded_letters)
                    print(f"\n✓ ผลลัพธ์: {result_text}\n")
                    recorded_letters = []
            elif key == ord('p'):
                print('Recognized words:', recognized_words)
    finally:
        landmarker.close()


def _list_image_files(dir_path: str):
    """Return a sorted list of supported image paths from a directory."""

    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.isdir(dir_path):
        return []

    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(extensions)
    )


def run_on_images(dir_path: str, max_num_hands: int, min_detection_confidence: float, model: Optional[str] = None):
    """Run gesture recognition on all images in a directory.

    Images are matched against an expected label derived from the filename (without extension).
    """

    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"สร้างโฟลเดอร์ใหม่สำหรับรูปภาพ: {dir_path}")
            print("กรุณาวางไฟล์รูป (jpg/png/...) ลงในโฟลเดอร์นี้ แล้วเรียกสคริปต์อีกครั้ง")
        except Exception as e:
            print(f"ไม่สามารถสร้างโฟลเดอร์: {dir_path} ({e})")
        return

    # Collect supported image files
    paths = _list_image_files(dir_path)

    if not paths:
        print(f"ไม่พบไฟล์ภาพในโฟลเดอร์: {dir_path}")
        return

    output_dir = os.path.join(dir_path, 'out')
    os.makedirs(output_dir, exist_ok=True)

    # Choose which MediaPipe API is available
    use_solutions = False
    try:
        import mediapipe as mp
        use_solutions = hasattr(mp, 'solutions')
    except Exception:
        use_solutions = False

    if not use_solutions and not model:
        print('No MediaPipe solutions API found and no --model provided (for tasks API).')
        return

    if not use_solutions and model is None:
        model = download_model(DEFAULT_TASK_MODEL_URL)
        print('Downloaded model to', model)

    correct = 0
    total = 0

    for path in paths:
        total += 1
        image = cv2.imread(path)
        if image is None:
            print(f"ไม่สามารถอ่านรูป: {path}")
            continue

        predicted = None
        if use_solutions:
            try:
                import mediapipe as mp
                mp_hands = mp.solutions.hands
                with mp_hands.Hands(
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5,
                ) as hands:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    res = hands.process(rgb)
                    if res and getattr(res, 'multi_hand_landmarks', None):
                        predicted = classify_gesture(res.multi_hand_landmarks[0].landmark)
            except Exception as e:
                print('Error running MediaPipe solutions on', path, ':', e)
        else:
            try:
                import mediapipe as mp
                from mediapipe.tasks.python.vision import hand_landmarker as hl_module
                from mediapipe.tasks.python.core.base_options import BaseOptions
                from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

                base_options = BaseOptions(model_asset_path=model)
                options = HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5,
                )

                landmarker = HandLandmarker.create_from_options(options)
                try:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
                    if hasattr(landmarker, 'detect'):
                        result = landmarker.detect(mp_image)
                    else:
                        result = landmarker.detect_for_video(mp_image, 0)
                    if result and getattr(result, 'hand_landmarks', None):
                        predicted = classify_gesture(result.hand_landmarks[0])
                finally:
                    landmarker.close()
            except Exception as e:
                print('Error running MediaPipe tasks on', path, ':', e)

        expected = os.path.splitext(os.path.basename(path))[0]
        ok = predicted == expected
        if ok:
            correct += 1

        label = f"predicted: {predicted or '---'} | expected: {expected} {'✓' if ok else '✗'}"
        print(f"{os.path.basename(path)} -> {label}")

        # Save output image with overlay (for review / training)
        try:
            overlay = image.copy()
            cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(out_path, overlay)
        except Exception as e:
            print(f"ไม่สามารถบันทึกภาพผลลัพธ์ได้: {e}")

    print(f"\nสรุป: ถูก {correct}/{total} ({correct/total*100:.1f}% )")
    print(f"บันทึกภาพผลลัพธ์ไว้ที่: {output_dir}")


def main(camera_index: int = 0, video_path: Optional[str] = None, max_num_hands: int = 2, min_detection_confidence: float = 0.5, model: Optional[str] = None, pic_dir: Optional[str] = None):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    if not _PIL_AVAILABLE:
        print('คำเตือน: Pillow ยังไม่ติดตั้ง; ข้อความไทยอาจแสดงเป็น ???? (ติดตั้งด้วย pip install pillow)')

    # หากไม่ได้ระบุ --pic-dir ให้ใช้โฟลเดอร์ 'pic' ในไดเรกทอรีของสคริปต์
    if pic_dir is None:
        pic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pic')

    # ถ้าเจอรูปในโฟลเดอร์ pic ให้รันโหมดประเมินจากรูป
    if _list_image_files(pic_dir):
        run_on_images(pic_dir, max_num_hands, min_detection_confidence, model=model)
        return

    cap = open_capture(camera_index, video_path=video_path)
    if not cap.isOpened():
        print("Failed to open camera or video source. Make sure a webcam is connected or provide a valid --video file.")
        return

    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            run_with_solutions(cap, max_num_hands, min_detection_confidence)
            cap.release()
            cv2.destroyAllWindows()
            return
    except Exception:
        pass

    try:
        import mediapipe as mp
        model_path = model
        if not model_path:
            print('No task model provided; downloading default model...')
            model_path = download_model(DEFAULT_TASK_MODEL_URL)
            print('Downloaded model to', model_path)

        run_with_tasks(cap, model_path, max_num_hands, min_detection_confidence)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def test_mode(max_num_hands: int = 2, min_detection_confidence: float = 0.5, model: Optional[str] = None):
    """Run a headless check to detect which MediaPipe API is available and exercise it without camera."""
    import numpy as np
    print('Running headless test...')
    try:
        import mediapipe as mp
        print('mediapipe module:', getattr(mp, '__file__', 'builtin'))
        if hasattr(mp, 'solutions'):
            print('Using mp.solutions (Hands)')
            try:
                mp_hands = mp.solutions.hands
                with mp_hands.Hands(
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=0.5,
                ) as hands:
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    img = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
                    img.flags.writeable = False
                    res = hands.process(img)
                    print('mp.solutions.Hands.process() returned:', bool(res and getattr(res, 'multi_hand_landmarks', None)))
            except Exception as e:
                print('Error exercising mp.solutions.Hands:', e)
        elif hasattr(mp, 'tasks'):
            print('mp.solutions not present; mediapipe.tasks detected')
            try:
                from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
                print('HandLandmarker class available; creating instance requires a .task model file (not attempted in --test)')
            except Exception as e:
                print('Error inspecting mediapipe.tasks APIs:', e)
        else:
            print('mediapipe installed but no recognizable API found (neither solutions nor tasks)')
    except Exception as e:
        print('mediapipe import failed:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand detection with MediaPipe (solutions or tasks) and OpenCV')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, default=None, help='Path to a video file (optional)')
    parser.add_argument('--max-hands', type=int, default=2, help='Maximum number of hands to detect')
    parser.add_argument('--min-detect-confidence', type=float, default=0.5, help='Min detection confidence')
    parser.add_argument('--model', type=str, default=None, help='Path to .task model for mediapipe tasks (optional)')
    parser.add_argument('--pic-dir', type=str, default=None, help='Path to folder containing images for training/evaluation (e.g., pic/)')
    parser.add_argument('--test', action='store_true', help='Run headless test (no camera)')
    args = parser.parse_args()

    if args.test:
        test_mode(max_num_hands=args.max_hands, min_detection_confidence=args.min_detect_confidence, model=args.model)
    else:
        main(
            camera_index=args.camera,
            video_path=args.video,
            max_num_hands=args.max_hands,
            min_detection_confidence=args.min_detect_confidence,
            model=args.model,
            pic_dir=args.pic_dir,
        )
