import argparse
import time
import tempfile
import urllib.request
import os
from typing import Optional

import cv2


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

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                    )

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Hand Detection (press q to quit)', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break


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
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Exiting.")
                break

            # convert BGR->RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # create mediapipe Image and run detect_for_video
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            image_out = frame.copy()
            h, w, _ = image_out.shape

            # draw landmarks if present
            if result and getattr(result, 'hand_landmarks', None):
                for hand_landmarks in result.hand_landmarks:
                    pts = []
                    for lm in hand_landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        pts.append((x, y))
                        cv2.circle(image_out, (x, y), 3, (0, 255, 0), -1)

                    # draw connections if available
                    try:
                        connections = hl_module.HandLandmarksConnections.HAND_CONNECTIONS
                        for conn in connections:
                            start = (int(hand_landmarks[conn.start].x * w), int(hand_landmarks[conn.start].y * h))
                            end = (int(hand_landmarks[conn.end].x * w), int(hand_landmarks[conn.end].y * h))
                            cv2.line(image_out, start, end, (0, 255, 255), 2)
                    except Exception:
                        pass

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(image_out, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Hand Detection (press q to quit)', image_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        landmarker.close()


def main(camera_index: int = 0, max_num_hands: int = 2, min_detection_confidence: float = 0.5, model: Optional[str] = None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    # try solutions first
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            run_with_solutions(cap, max_num_hands, min_detection_confidence)
            cap.release()
            cv2.destroyAllWindows()
            return
    except Exception:
        pass

    # fallback: tasks API
    try:
        import mediapipe as mp
        # ensure model exists
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
                # Can only check presence of Task classes without model
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
    parser.add_argument('--max-hands', type=int, default=2, help='Maximum number of hands to detect')
    parser.add_argument('--min-detect-confidence', type=float, default=0.5, help='Min detection confidence')
    parser.add_argument('--model', type=str, default=None, help='Path to .task model for mediapipe tasks (optional)')
    parser.add_argument('--test', action='store_true', help='Run headless test (no camera)')
    args = parser.parse_args()

    if args.test:
        test_mode(max_num_hands=args.max_hands, min_detection_confidence=args.min_detect_confidence, model=args.model)
    else:
        main(camera_index=args.camera, max_num_hands=args.max_hands, min_detection_confidence=args.min_detect_confidence, model=args.model)
