import mediapipe as mp

print(f"mediapipe version: {mp.__version__}")
print(f"has solutions: {hasattr(mp, 'solutions')}")

if hasattr(mp, 'solutions'):
    print("solutions found!")
    print(f"has hands: {hasattr(mp.solutions, 'hands')}")
else:
    print("ERROR: mediapipe does not have 'solutions' attribute")
    print(f"Available attributes: {dir(mp)}")
