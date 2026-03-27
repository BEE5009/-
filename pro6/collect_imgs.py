"""Collect training data for sign language detection"""
import os
import cv2
import numpy as np
import time
import sys
from PIL import Image, ImageDraw, ImageFont

# Force stdout/stderr to UTF-8 on Windows so emojis/Thai text don't crash with cp1252
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define English letters (A-Z = 26 classes)
NUMBER_OF_CLASSES = 26
DATASET_SIZE = 100
COUNTDOWN_SECONDS = 3  # นับถอยหลังก่อนจับภาพ

# Global variables for image adjustment
brightness_value = 0
contrast_value = 50
is_camera_on = True


def safe_read_frame(cap, retries=3):
    """อ่านเฟรมจากกล้อง ป้องกัน frame เป็น None และจัดการ exception ของ OpenCV"""
    if cap is None or not cap.isOpened():
        return False, None

    for attempt in range(retries):
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            print(f"⚠️ cv2.read failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(0.05)
            continue

        if not ret or frame is None or frame.size == 0:
            print(f"⚠️ ได้ frame ไม่ถูกต้อง (attempt {attempt+1}/{retries})")
            time.sleep(0.05)
            continue

        return True, frame

    return False, None


def adjust_brightness_contrast(frame, brightness=0, contrast=50):
    """ปรับความสว่างและ contrast ของภาพ"""
    # Adjust brightness
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness

    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow

    buf = cv2.convertScaleAbs(frame, alpha=alpha_b, beta=gamma_b)

    # Adjust contrast
    contrast_alpha = float(contrast) / 50
    buf = cv2.convertScaleAbs(buf, alpha=contrast_alpha, beta=0)

    return np.clip(buf, 0, 255).astype(np.uint8)


def put_text_unicode(frame, text, pos, font_size=24, color=(0, 255, 0), thickness=1):
    """วาดข้อความ Unicode (Thai/Emoji) ด้วย PIL แล้วแปลงกลับ OpenCV"""
    # หา font ไทยจากโฟลเดอร์ฟอนต์ในโปรเจ็กต์ หรือ fallback system
    font_path_candidates = [
        "./ฟอนต์/THSarabunPSKv1.0/Fonts TH SarabunPSK v1.0/THSarabunPSK.ttf",
        "./ฟอนต์/THSarabunNew/THSarabunNew.ttf",
        "./ฟอนต์/thai/THSarabunNew/THSarabunNew.ttf"
    ]
    font = None
    for fp in font_path_candidates:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                font = None
    if font is None:
        font = ImageFont.load_default()

    # แปลง frame เป็น RGB เพื่อ PIL ใช้งานอย่างถูกต้อง แล้วกลับเป็น BGR สำหรับ OpenCV
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)
    r, g, b = color
    draw.text(pos, text, font=font, fill=(r, g, b))
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return frame_bgr


def enhance_image_visibility(frame):
    """ปรับปรุงความชัดของภาพโดย CLAHE"""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced


def draw_countdown(frame, countdown_value, text="กำลังจับภาพใน..."):
    """วาดนับถอยหลังบนภาพ"""
    h, w = frame.shape[:2]
    
    # พื้นหลังโปร่งแสง
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 150, h//2 - 100), (w//2 + 150, h//2 + 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # ข้อความ
    frame = put_text_unicode(frame, text, (w//2 - 140, h//2 - 40), font_size=30, color=(0, 255, 0))
    
    # ตัวเลขนับถอยหลัง (ใหญ่มาก)
    frame = put_text_unicode(frame, str(countdown_value), (w//2 - 40, h//2 + 20), font_size=70, color=(0, 255, 255))
    
    return frame


def draw_collection_status(frame, current_count, total_count, elapsed_time):
    """แสดงสถานะการจับภาพ"""
    h, w = frame.shape[:2]
    
    # ข้อมูลบน
    progress_text = f"จับภาพ: {current_count}/{total_count}"
    frame = put_text_unicode(frame, progress_text, (10, 40), font_size=28, color=(0,255,0))
    
    # เวลา
    text_time = f"เวลา: {elapsed_time:.1f}s"
    frame = put_text_unicode(frame, text_time, (10, 90), font_size=28, color=(0,255,255))
    
    # Progress bar
    bar_width = 300
    bar_height = 30
    bar_x = (w - bar_width) // 2
    bar_y = h - 60
    
    # พื้นหลัง
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    
    # filled portion
    if total_count > 0:
        filled_width = int(bar_width * current_count / total_count)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
    
    # text
    percent_text = f"{(current_count/total_count*100):.0f}%"
    frame = put_text_unicode(frame, percent_text, (w//2 - 30, bar_y + 14), font_size=24, color=(255,255,255))
    
    return frame


def draw_controls_info(frame):
    """แสดงปุ่มควบคุม"""
    h, w = frame.shape[:2]
    
    controls = [
        "Q=ออก | B/V=ความสว่าง | C/X=Contrast",
        "SPACE=เปิด/ปิดกล้อง"
    ]
    
    for i, line in enumerate(controls):
        frame = put_text_unicode(frame, line, (10, h - 50 + i*30), font_size=20, color=(255,255,0))
    
    return frame


def wait_for_ready(cap, letter):
    """รอให้ผู้ใช้พร้อม พร้อมแสดง countdown"""
    global brightness_value, contrast_value, is_camera_on
    
    print(f'\n🖐️ ตัวอักษร {letter} - กำลังรอให้พร้อม...')
    
    start_time = time.time()
    while True:
        if not is_camera_on:
            ret = True
            frame = np.zeros((480, 720, 3), dtype=np.uint8)
            frame = put_text_unicode(frame, "กล้องปิด - กด SPACE เพื่อเปิด", (150, 240), font_size=28, color=(0,0,255))
        else:
            ret, frame = safe_read_frame(cap)
        
        if not ret:
            print("⚠️ ข้อผิดพลาด: ไม่สามารถอ่านกล้อง")
            cv2.waitKey(100)
            continue
        
        # ปรับปรุงภาพ
        if is_camera_on:
            frame = enhance_image_visibility(frame)
            frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)
        
        # แสดงข้อมูล
        h, w = frame.shape[:2]
        text = f'พร้อมจับตัวอักษร "{letter}"? กด Q เพื่อเริ่ม'
        frame = put_text_unicode(frame, text, (30, 50), font_size=26, color=(0,255,0))
        
        elapsed = time.time() - start_time
        frame = put_text_unicode(frame, f"เวลา: {elapsed:.1f}s", (30, 100), font_size=24, color=(0,255,255))
        
        frame = draw_controls_info(frame)
        
        cv2.imshow('จับภาพตัวอักษร', frame)
        key = cv2.waitKey(25) & 0xFF
        
        if key == ord('q'):
            print("✓ เริ่มจับภาพ!")
            break
        elif key == ord('b'):
            brightness_value = min(100, brightness_value + 5)
            print(f"  💡 ความสว่าง: {brightness_value}")
        elif key == ord('v'):
            brightness_value = max(-100, brightness_value - 5)
            print(f"  💡 ความสว่าง: {brightness_value}")
        elif key == ord('c'):
            contrast_value = min(100, contrast_value + 5)
            print(f"  🎨 Contrast: {contrast_value}")
        elif key == ord('x'):
            contrast_value = max(0, contrast_value - 5)
            print(f"  🎨 Contrast: {contrast_value}")
        elif key == ord(' '):
            is_camera_on = not is_camera_on
            if is_camera_on:
                print("  ✓ เปิดกล้อง")
            else:
                print("  ✗ ปิดกล้อง")


def collect_samples(cap, class_dir, letter):
    """จับภาพแต่ละตัวอักษร พร้อมนับถอยหลังและแสดงเวลา"""
    global brightness_value, contrast_value, is_camera_on
    
    print(f'📸 กำลังจับภาพตัวอักษร {letter}...')
    
    counter = 0
    total_start_time = time.time()
    
    while counter < DATASET_SIZE:
        if not is_camera_on:
            print("  ⏸️ กล้องปิด - กด SPACE เพื่อเปิด")
            cv2.waitKey(100)
            continue
        
        # === นับถอยหลัง ===
        countdown_start = time.time()
        while time.time() - countdown_start < COUNTDOWN_SECONDS:
            ret, frame = safe_read_frame(cap)
            if not ret:
                print("⚠️ ข้อผิดพลาด: ไม่สามารถอ่านกล้อง")
                cv2.waitKey(100)
                continue
            
            # ปรับปรุงภาพ
            frame = enhance_image_visibility(frame)
            frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)
            
            # นับถอยหลัง
            remaining = COUNTDOWN_SECONDS - (time.time() - countdown_start)
            countdown_val = max(1, int(remaining) + 1)
            frame = draw_countdown(frame, countdown_val, f"จับภาพรูปที่ {counter + 1}")
            
            elapsed_total = time.time() - total_start_time
            frame = draw_collection_status(frame, counter, DATASET_SIZE, elapsed_total)
            frame = draw_controls_info(frame)
            
            cv2.imshow('จับภาพตัวอักษร', frame)
            key = cv2.waitKey(25) & 0xFF
            
            if key == ord('q'):
                print(f"  ⏹️ หยุดจับ - รวม {counter} รูป")
                return True
            elif key == ord('b'):
                brightness_value = min(100, brightness_value + 5)
            elif key == ord('v'):
                brightness_value = max(-100, brightness_value - 5)
            elif key == ord('c'):
                contrast_value = min(100, contrast_value + 5)
            elif key == ord('x'):
                contrast_value = max(0, contrast_value - 5)
            elif key == ord(' '):
                is_camera_on = not is_camera_on
    
        ret, frame = safe_read_frame(cap)
        if not ret:
            print("⚠️ ข้อผิดพลาด: ไม่สามารถอ่านกล้อง")
            cv2.waitKey(100)
            continue
    
        frame = enhance_image_visibility(frame)
        frame = adjust_brightness_contrast(frame, brightness_value, contrast_value)

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2 - 100, h//2 - 80), (w//2 + 100, h//2 + 80), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        frame = put_text_unicode(frame, "SAVED! ✓", (w//2 - 90, h//2 + 10), font_size=36, color=(0,255,0))

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        elapsed_total = time.time() - total_start_time
        frame = draw_collection_status(frame, counter, DATASET_SIZE, elapsed_total)
        frame = draw_controls_info(frame)

        cv2.imshow('จับภาพตัวอักษร', frame)
        key = cv2.waitKey(500) & 0xFF

        if key == ord('q'):
            print(f"  ⏹️ หยุดจับ - รวม {counter} รูป")
            return True
        elif key == ord('b'):
            brightness_value = min(100, brightness_value + 5)
        elif key == ord('v'):
            brightness_value = max(-100, brightness_value - 5)
        elif key == ord('c'):
            contrast_value = min(100, contrast_value + 5)
        elif key == ord('x'):
            contrast_value = max(0, contrast_value - 5)
        elif key == ord(' '):
            is_camera_on = not is_camera_on
    
    total_elapsed = time.time() - total_start_time
    print(f'✓ เก็บ {counter} รูป ใช้เวลา {total_elapsed:.1f} วินาที')
    return True


def main():
    """โปรแกรมหลัก"""
    global is_camera_on
    
    print("\n" + "="*60)
    print("🖐️ ระบบรวมข้อมูลสำหรับตรวจจับท่ามือ")
    print("="*60)
    print("ตัวอักษรที่จะจับ: A-Z (26 ตัว)")
    print(f"จำนวนรูปต่อตัวอักษร: {DATASET_SIZE} รูป")
    print(f"เวลาอยู่ระหว่างจับ: {COUNTDOWN_SECONDS} วินาที")
    print("\n⌨️ ปุ่มควบคุม:")
    print("   Q = ออก/ข้ามตัวอักษร")
    print("   B/V = เพิ่ม/ลดความสว่าง")
    print("   C/X = เพิ่ม/ลด Contrast")
    print("   SPACE = เปิด/ปิดกล้อง")
    print("="*60 + "\n")
    
    cap = None
    camera_idx = -1
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L]

    for idx in range(5):
        for backend in backends:
            cap_try = cv2.VideoCapture(idx, backend)
            if not cap_try.isOpened():
                cap_try.release()
                continue

            ret, frame = safe_read_frame(cap_try)
            if ret:
                cap = cap_try
                camera_idx = idx
                print(f"✓ เปิดกล้องสำเร็จ (Camera {idx}, backend={backend})\n")
                break

            cap_try.release()

        if cap is not None:
            break

    if cap is None:
        print("✗ ไม่สามารถเปิดกล้องได้ - ตรวจสอบ:")
        print("  - กล้องเสียบอยู่หรือไม่?")
        print("  - โปรแกรมอื่นใช้กล้องอยู่หรือไม่?")
        print("  - ไดร์เวอร์กล้องทำงานปกติหรือไม่?")
        print("\n💡 หรือลองใช้วิดีโอทดสอบแทน:")
        print("  - เปลี่ยน: cv2.VideoCapture(0)")
        print("  - เป็น: cv2.VideoCapture('video.mp4')")
        return
    
    is_camera_on = True
     
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    overall_start_time = time.time()

    for j in range(NUMBER_OF_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        letter = chr(ord('A') + j)
        
        wait_for_ready(cap, letter)
        
        success = collect_samples(cap, class_dir, letter)
        
        if not success:
            break

    cap.release()
    cv2.destroyAllWindows()

    overall_elapsed = time.time() - overall_start_time
    
    print("\n" + "="*60)
    print(f"✓ เสร็จสิ้น!")
    print(f"  ใช้เวลารวม: {overall_elapsed:.1f} วินาที ({overall_elapsed/60:.1f} นาที)")
    print(f"  บันทึกข้อมูลไว้ใน: {DATA_DIR}")
    print("="*60 + "\n")

if __name__ == "__main__":
    is_camera_on = True
    main()
