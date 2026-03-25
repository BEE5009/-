"""verify_env.py
Simple helper to print Python executable and key package versions for system/venv checks.
"""
import sys
import importlib
import traceback

def show(name):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, '__version__', None)
        path = getattr(m, '__file__', 'builtin')
        print(f"{name}: version={ver}, file={path}")
    except Exception as e:
        print(f"{name}: ERROR: {e}")

def main():
    print('PYTHON:', sys.executable)
    print('sys.path[0]:', sys.path[0])
    show('numpy')
    show('cv2')
    try:
        import mediapipe as mp
        print('mediapipe.__file__:', getattr(mp,'__file__',None))
        print('mediapipe has solutions:', hasattr(mp, 'solutions'))
        print('mediapipe has tasks:', hasattr(mp, 'tasks'))
    except Exception as e:
        print('mediapipe: ERROR')
        traceback.print_exc()

    # quick import of our script (no execution)
    try:
        sys.path.insert(0, '.')
        import hand_detection
        print('hand_detection import: OK')
    except Exception as e:
        print('hand_detection import: ERROR')
        traceback.print_exc()

if __name__ == '__main__':
    main()
