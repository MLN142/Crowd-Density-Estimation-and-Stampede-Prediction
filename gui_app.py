import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import threading
from torchvision import transforms
from models.csrnet import CSRNet
from ultralytics import YOLO
import scipy.ndimage
import pygame # For audio alerts

class CrowdCountingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Counting Application")
        self.root.geometry("1400x800")
        self.root.resizable(True, True)
        
        # Variables
        self.model_choice = tk.StringVar(value="csrnet")
        self.input_choice = tk.StringVar(value="file")
        self.video_path = tk.StringVar(value="")
        self.is_processing = False
        self.cap = None
        self.current_frame = None
        self.threshold_value = tk.DoubleVar(value=0.022)  # Default CSRNet threshold
        self.count_threshold = tk.IntVar(value=100)  # Default count threshold for alerts
        self.alert_active = False  # Track if alert is currently showing
        self.flash_state = False # Track flash state for color toggling
        
        # Model paths
        self.csrnet_path = "best_csrnet.pth"
        self.yolo_path = "best.pt"
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models (loaded on demand)
        self.csrnet_model = None
        self.yolo_model = None
        
        # CSRNet transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Fixed dimensions
        self.fixed_width = 1280
        self.fixed_height = 720
        
        # Initialize Audio
        try:
            pygame.mixer.init()
            self.sound_enabled = True
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            self.sound_enabled = False

        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model Selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Select Model:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="CSRNet (Density Map)", variable=self.model_choice, value="csrnet").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="YOLOv8 (Object Detection)", variable=self.model_choice, value="yolo").pack(side=tk.LEFT, padx=5)
        
        # Input Selection
        input_frame = ttk.Frame(control_frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Select Input:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(input_frame, text="Video File", variable=self.input_choice, value="file", command=self.on_input_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(input_frame, text="Camera", variable=self.input_choice, value="camera", command=self.on_input_change).pack(side=tk.LEFT, padx=5)
        
        # File Selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Video File:").pack(side=tk.LEFT, padx=5)
        self.file_entry = ttk.Entry(file_frame, textvariable=self.video_path, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        self.browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        # CSRNet Threshold Slider
        self.threshold_frame = ttk.Frame(control_frame)
        self.threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.threshold_frame, text="CSRNet Threshold:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.threshold_slider = ttk.Scale(self.threshold_frame, from_=0.001, to=0.100, variable=self.threshold_value, orient=tk.HORIZONTAL, length=200)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        self.threshold_label = ttk.Label(self.threshold_frame, text=f"{self.threshold_value.get():.3f}")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Update threshold label when slider moves
        def update_threshold_label(*args):
            self.threshold_label.config(text=f"{self.threshold_value.get():.3f}")
        self.threshold_value.trace_add('write', update_threshold_label)
        
        # Count Threshold Input
        count_threshold_frame = ttk.Frame(control_frame)
        count_threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(count_threshold_frame, text="Count Alert Threshold:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.count_threshold_entry = ttk.Entry(count_threshold_frame, textvariable=self.count_threshold, width=10)
        self.count_threshold_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(count_threshold_frame, text="(Alert when count exceeds this value)").pack(side=tk.LEFT, padx=5)
        
        # Control Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(button_frame, text="Start Processing", command=self.start_processing, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Exit", command=self.on_closing, width=15).pack(side=tk.LEFT, padx=5)
        
        # Alert Frame (dedicated space for alert)
        self.alert_frame = ttk.Frame(main_frame)
        self.alert_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Alert Label (Always visible but empty when inactive)
        self.alert_label = tk.Label(self.alert_frame, text="", 
                                     font=("Arial", 14, "bold"), 
                                     foreground="black", 
                                     background=self.root.cget("bg"),
                                     height=2) # Fixed height to reserve space
        self.alert_label.pack(fill=tk.X)

        # Video Display
        display_frame = ttk.LabelFrame(main_frame, text="Video Display", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas = tk.Canvas(display_frame, width=1280, height=720, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        # Status Bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: -", relief=tk.SUNKEN, anchor=tk.E, width=15)
        self.fps_label.pack(side=tk.RIGHT)
        

        
    def on_input_change(self):
        if self.input_choice.get() == "camera":
            self.file_entry.config(state=tk.DISABLED)
            self.browse_btn.config(state=tk.DISABLED)
            self.video_path.set("")
        else:
            self.file_entry.config(state=tk.NORMAL)
            self.browse_btn.config(state=tk.NORMAL)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if filename:
            self.video_path.set(filename)
    
    def load_model(self):
        try:
            if self.model_choice.get() == "csrnet":
                if self.csrnet_model is None:
                    self.update_status("Loading CSRNet model...")
                    self.csrnet_model = CSRNet().to(self.device)
                    self.csrnet_model.load_state_dict(torch.load(self.csrnet_path, map_location=self.device))
                    self.csrnet_model.eval()
                    self.update_status("CSRNet model loaded successfully")
            else:
                if self.yolo_model is None:
                    self.update_status("Loading YOLOv8 model...")
                    self.yolo_model = YOLO(self.yolo_path)
                    self.update_status("YOLOv8 model loaded successfully")
            return True
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            return False
    
    def start_processing(self):
        # Validate input
        if self.input_choice.get() == "file" and not self.video_path.get():
            messagebox.showwarning("Input Required", "Please select a video file")
            return
        
        # Load model
        if not self.load_model():
            return
        
        # Open video source
        if self.input_choice.get() == "camera":
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.video_path.get())
        
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", "Failed to open video source")
            return
        
        # Update UI
        self.is_processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        self.update_status("Processing stopped")
    
    def resize_with_padding(self, image):
        old_height, old_width = image.shape[:2]
        scale = min(self.fixed_width / old_width, self.fixed_height / old_height)
        new_width = int(old_width * scale)
        new_height = int(old_height * scale)
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.fixed_width - new_width) // 2
        pad_h = (self.fixed_height - new_height) // 2
        padded_img = np.zeros((self.fixed_height, self.fixed_width, 3), dtype=image.dtype)
        padded_img[pad_h:pad_h+new_height, pad_w:pad_w+new_width, :] = resized_img
        return padded_img
    
    def process_csrnet(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = self.resize_with_padding(img_rgb)
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.csrnet_model(img_tensor)
            density_map = output.squeeze().cpu().numpy()
        
        # Post-processing
        density_map = scipy.ndimage.median_filter(density_map, size=3)
        density_map = np.clip(density_map, 0, None)
        # Use threshold from slider
        threshold = self.threshold_value.get()
        density_map[density_map < threshold] = 0
        
        count = np.sum(density_map)
        
        # Create visualization
        normalized = density_map.copy()
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
        normalized = (normalized * 255).astype(np.uint8)
        normalized_resized = cv2.resize(normalized, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(normalized_resized, cv2.COLORMAP_JET)
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
        
        # Add count text
        cv2.putText(overlay, f"Count: {count:.1f}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        return overlay, count
    
    def process_yolo(self, frame):
        frame = self.resize_with_padding(frame)
        results = self.yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes.xyxy, 'cpu') else results[0].boxes.xyxy
        
        count = len(boxes)
        
        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add count text
        cv2.putText(frame, f"Count: {count}", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        
        return frame, count
    
    def process_video(self):
        import time
        frame_count = 0
        start_time = time.time()
        
        while self.is_processing:
            ret, frame = self.cap.read()
            if not ret:
                if self.input_choice.get() == "file":
                    # Video ended, loop back
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Process frame based on model
            try:
                if self.model_choice.get() == "csrnet":
                    processed_frame, count = self.process_csrnet(frame)
                else:
                    processed_frame, count = self.process_yolo(frame)
                
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Check count threshold and show alert if exceeded
                threshold = self.count_threshold.get()
                # print(f"Count: {count:.1f}, Threshold: {threshold}, Alert Active: {self.alert_active}")  # Debug
                if count > threshold:
                    if not self.alert_active:
                        self.root.after(0, self.show_alert)
                        self.alert_active = True
                else:
                    if self.alert_active:
                        self.root.after(0, self.hide_alert)
                        self.alert_active = False
                
                # Update display
                self.display_frame(processed_frame)
                self.root.after(0, self.update_status, f"Processing | Model: {self.model_choice.get().upper()} | Count: {count:.1f}")
                self.root.after(0, self.fps_label.config, {"text": f"FPS: {fps:.1f}"})
                
            except Exception as e:
                print(f"Processing error: {e}")
                continue
        
        if self.cap:
            self.cap.release()
        self.root.after(0, self.stop_btn.config, {"state": tk.DISABLED})
        self.root.after(0, self.start_btn.config, {"state": tk.NORMAL})
    
    def display_frame(self, frame):
        # Convert to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk
    
    def show_alert(self):
        """Show the alert label when count threshold is exceeded"""
        try:
            if self.alert_label.cget("text") == "":
                self.alert_label.config(text="⚠️ ALERT: Count Threshold Exceeded! ⚠️")
                # print("Alert shown!")  # Debug
                # Start flashing
                self.flash_alert()
                
                # Play sound
                if self.sound_enabled:
                    try:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.load("beep.mp3")
                            pygame.mixer.music.play(-1) # Loop indefinitely
                    except Exception as e:
                        print(f"Error playing sound: {e}")
        except Exception as e:
            print(f"Error showing alert: {e}")
    
    def hide_alert(self):
        """Hide the alert label when count is below threshold"""
        try:
            if self.alert_label.cget("text") != "":
                self.alert_label.config(text="", background=self.root.cget("bg"), foreground="black")
                # print("Alert hidden!")  # Debug
                
                # Stop sound
                if self.sound_enabled:
                    pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error hiding alert: {e}")
    
    def flash_alert(self):
        """Make the alert flash to get attention"""
        if self.alert_active:
            self.flash_state = not self.flash_state
            
            if self.flash_state:
                new_bg = "red"
                new_fg = "white"
            else:
                new_bg = "yellow"
                new_fg = "black"
            
            self.alert_label.config(background=new_bg, foreground=new_fg)
            self.root.after(500, self.flash_alert)  # Flash every 500ms
    
    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
    
    def on_closing(self):
        if self.is_processing:
            self.stop_processing()
        if self.sound_enabled:
            pygame.mixer.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CrowdCountingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
