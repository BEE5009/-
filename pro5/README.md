# Hand Detection with Python (MediaPipe + OpenCV)

คำอธิบายสั้น ๆ: สคริปต์ `hand_detection.py` ใช้กล้องเพื่อตรวจจับมือแบบเรียลไทม์ด้วย MediaPipe Hands และแสดงจุด landmark บนหน้าจอ รวมถึงจำแนกท่าทางมือเป็นคำภาษาไทยและเก็บคำที่ตรวจจับได้


การติดตั้ง (แนะนำให้ใช้ `virtualenv` และรันใน Windows)

1) สร้างและเปิดใช้งาน venv (cmd):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

2) ติดตั้ง dependencies (ไฟล์ `requirements.txt` ถูกปักเวอร์ชันไว้แล้ว):

```cmd
pip install -r requirements.txt
```

> หากแสดงผลตัวอักษรไทยเป็น `?????` ให้แน่ใจว่า `pillow` ถูกติดตั้ง (มันจะช่วยวาดข้อความ Unicode บน OpenCV)
3) เรียกใช้งาน (มี wrapper ช่วยเรียกใน Windows):

```cmd
run_windows.bat
```

หรือโดยตรง (เรียก interpreter ของ venv):

```cmd
"d:\code alllll\python\.venv\Scripts\python.exe" "d:\code alllll\python\hand_detection.py"
```

หมายเหตุการดีบักและการแก้ปัญหา
- หากเจอ AttributeError เกี่ยวกับ `mediapipe.solutions`: ให้รัน `verify_env.py` ทั้งด้วย system Python และ venv Python เพื่อเปรียบเทียบ (ตัวอย่าง):

```cmd
"D:\python\python 3.13.9\python.exe" verify_env.py
"d:\code alllll\python\.venv\Scripts\python.exe" verify_env.py
```

- หาก `numpy` ขึ้น error ให้แน่ใจว่าใช้เวอร์ชันที่ตรงกับ `opencv-python` (ไฟล์ `requirements.txt` ปักไว้แล้ว)
- หากต้องการใช้ `mediapipe.tasks` แทน `mp.solutions` สคริปต์ `hand_detection.py` จะเลือกอัตโนมัติ (fallback)

การตั้งค่า VS Code
- เปิด Command Palette -> `Python: Select Interpreter` -> เลือก interpreter ที่ชี้ไปที่ `.venv` ของโปรเจค
- ตรวจสอบ `launch.json` ถ้าต้องการรัน/ดีบัก ให้แน่ใจว่า `python` path อยู่ใน `python` setting ของ workspace


กด `q` หรือ `Esc` เพื่อออกจากโปรแกรม (และพิมพ์คำที่ตรวจจับได้ทั้งหมด)

กด `p` เพื่อพิมพ์คำที่ตรวจจับได้ในขณะนั้น

หมายเหตุ:
- หากต้องการปรับความไวการตรวจจับ ให้แก้ `--min-detect-confidence` เป็นค่าสูงขึ้นหรือต่ำลง
- หากต้องการใช้กล้องอื่น ให้เปลี่ยนพารามิเตอร์ `--camera` เป็นดัชนีกล้องที่ต้องการ
