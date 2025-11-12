# projectvai.py
import os, time, threading, queue, cv2, cvlib as cv, pytesseract, pyttsx3
from cvlib.object_detection import draw_bbox
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# -------------------- Speech Queue --------------------
speech_q = queue.Queue()
engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

def speech_worker():
    while True:
        text = speech_q.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Speech error:", e)

threading.Thread(target=speech_worker, daemon=True).start()

def speak_now(text):
    if text:
        speech_q.put(text)

# -------------------- Invision AI App --------------------
class InvisionAI:
    def __init__(self, root):
        self.root = root
        self.root.title("Invision AI — An Eye for the Blind")
        self.root.configure(bg="#0A0A0A")
        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

        # Header
        tk.Label(root, text="👁️ Invision AI", font=("Helvetica", 48, "bold"),
                 bg="#0A0A0A", fg="#00E0FF").pack(pady=(20, 0))
        tk.Label(root, text="An Eye for the Blind", font=("Helvetica", 22),
                 bg="#0A0A0A", fg="white").pack(pady=(0, 20))

        # Main frame split
        self.main_frame = tk.Frame(root, bg="#111111")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.camera_label = tk.Label(self.main_frame, bg="#000000")
        self.camera_label.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.info_box = tk.Text(self.main_frame, wrap="word", font=("Helvetica", 16),
                                bg="#111111", fg="#E0E0E0", relief="flat")
        self.info_box.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Buttons
        btn_frame = tk.Frame(root, bg="#0A0A0A")
        btn_frame.pack(pady=20)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 16, "bold"), padding=15)

        ttk.Button(btn_frame, text="🎯 Object Detection",
                   command=self.start_object_mode).grid(row=0, column=0, padx=15)
        ttk.Button(btn_frame, text="🗣 Text Reader",
                   command=self.start_text_mode).grid(row=0, column=1, padx=15)
        ttk.Button(btn_frame, text="🛑 Stop",
                   command=self.stop_mode).grid(row=0, column=2, padx=15)
        ttk.Button(btn_frame, text="❌ Exit",
                   command=self.exit_app).grid(row=0, column=3, padx=15)

        self.running = False
        self.mode = None
        self.cap = None
        self.log_file = os.path.expanduser("~/Documents/Invision_AI_Detection_Log.txt")

        # detection cooldowns
        self.obj_last, self.txt_last = "", ""
        self.obj_time, self.txt_time = 0, 0
        self.obj_delay, self.txt_delay = 1.0, 1.5  # lower delay = faster speech response

        speak_now("Welcome to Invision AI. An eye for the blind. Choose object detection or text reading mode.")
        self.log("Application started.")

    # -------------------- Logging --------------------
    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        self.info_box.insert("end", line)
        self.info_box.see("end")
        with open(self.log_file, "a") as f:
            f.write(line)

    # -------------------- Camera --------------------
    def get_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.cap

    def update_camera_feed(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((900, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

    # -------------------- Object Detection --------------------
    def start_object_mode(self):
        if self.running: return
        self.running = True
        self.mode = "object"
        self.log("Starting object detection...")
        speak_now("Object detection mode activated.")
        threading.Thread(target=self.object_loop, daemon=True).start()

    def object_loop(self):
        cap = self.get_camera()
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))
            try:
                bbox, labels, conf = cv.detect_common_objects(frame, confidence=0.35, model="yolov3-tiny")
                frame = draw_bbox(frame, bbox, labels, conf)
            except Exception as e:
                labels = []
                self.log(f"Detection error: {e}")

            detected = ", ".join(sorted(set(labels))) if labels else ""
            if detected and (detected != self.obj_last or time.time() - self.obj_time > self.obj_delay):
                self.obj_last = detected
                self.obj_time = time.time()
                self.log(f"Detected: {detected}")
                speak_now(f"I can see {detected}")

            self.update_camera_feed(frame)
            self.root.update_idletasks()
            time.sleep(0.05)

        if cap: cap.release()
        cv2.destroyAllWindows()

    # -------------------- Text Reader --------------------
    def start_text_mode(self):
        if self.running: return
        self.running = True
        self.mode = "text"
        self.log("Starting text reader...")
        speak_now("Text reading mode activated.")
        threading.Thread(target=self.text_loop, daemon=True).start()

    def text_loop(self):
        cap = self.get_camera()
        while self.running:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            text = pytesseract.image_to_string(gray).strip()
            clean = " ".join(text.split())
            if clean and len(clean) > 4 and (clean != self.txt_last or time.time() - self.txt_time > self.txt_delay):
                self.txt_last = clean
                self.txt_time = time.time()
                self.log(f"Text detected: {clean}")
                speak_now(f"The text says {clean}")

            self.update_camera_feed(frame)
            self.root.update_idletasks()
            time.sleep(0.1)

        if cap: cap.release()
        cv2.destroyAllWindows()

    # -------------------- Stop / Exit --------------------
    def stop_mode(self):
        if not self.running:
            self.log("No active mode to stop.")
            return
        self.running = False
        speak_now("Stopping current mode.")
        self.log("Mode stopped.")

    def exit_app(self):
        self.running = False
        speak_now("Invision AI shutting down. Goodbye.")
        self.log("Application closed.")
        speech_q.put(None)
        time.sleep(1)
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except:
            pass
        self.root.destroy()

# -------------------- Run --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = InvisionAI(root)
    root.mainloop()