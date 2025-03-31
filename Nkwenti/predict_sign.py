import sys

# Check for required modules
try:
    import cv2
    import numpy as np
    import tensorflow as tf
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    from PIL import Image, ImageTk
except ImportError as e:
    module_name = str(e).split("'")[1]
    if module_name == "PIL":
        print("Error: PIL module not found. Please install it using:")
        print("pip install pillow")
    else:
        print(f"Error: {module_name} module not found. Please install required dependencies:")
        print("pip install opencv-python numpy tensorflow pillow")
    sys.exit(1)

IMG_WIDTH = 30
IMG_HEIGHT = 30

class TrafficSignClassifier:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Traffic Sign Classifier")
        self.window.geometry("600x500")
        self.model = None
        self.setup_gui()

    def setup_gui(self):
        # Model selection
        model_frame = ttk.LabelFrame(self.window, text="Model Selection", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        self.model_path = tk.StringVar()
        ttk.Label(model_frame, text="Model:").pack(side="left")
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).pack(side="left", padx=5)
        ttk.Button(model_frame, text="Browse", command=self.load_model).pack(side="left")

        # Image selection
        image_frame = ttk.LabelFrame(self.window, text="Image Selection", padding=10)
        image_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(image_frame, text="Select Image", command=self.select_image).pack()
        
        # Image preview
        self.preview_label = ttk.Label(image_frame)
        self.preview_label.pack(pady=10)

        # Results
        results_frame = ttk.LabelFrame(self.window, text="Results", padding=10)
        results_frame.pack(fill="x", padx=10, pady=5)
        
        self.category_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.confidence_var = tk.StringVar()
        
        ttk.Label(results_frame, text="Category:").grid(row=0, column=0, sticky="w")
        ttk.Label(results_frame, textvariable=self.category_var).grid(row=0, column=1, sticky="w")
        
        ttk.Label(results_frame, text="Sign Name:").grid(row=1, column=0, sticky="w")
        ttk.Label(results_frame, textvariable=self.name_var).grid(row=1, column=1, sticky="w")
        
        ttk.Label(results_frame, text="Confidence:").grid(row=2, column=0, sticky="w")
        ttk.Label(results_frame, textvariable=self.confidence_var).grid(row=2, column=1, sticky="w")

    def load_model(self):
        filename = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if filename:
            self.model_path.set(filename)
            try:
                self.model = tf.keras.models.load_model(filename)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def select_image(self):
        if self.model is None:
            tk.messagebox.showerror("Error", "Please load a model first")
            return

        filename = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if filename:
            # Load and display preview
            preview = Image.open(filename)
            preview.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(preview)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo

            # Process image and make prediction
            try:
                img = load_image(filename)
                prediction = self.model.predict(img, verbose=0)
                category = np.argmax(prediction[0])
                confidence = prediction[0][category] * 100

                self.category_var.set(str(category))
                self.name_var.set(SIGN_CATEGORIES[category])
                self.confidence_var.set(f"{confidence:.2f}%")
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def run(self):
        self.window.mainloop()

def load_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Error: Could not load image {image_path}")
    
    # Resize image
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)
    return img

# Dictionary of traffic sign categories
SIGN_CATEGORIES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def main():
    app = TrafficSignClassifier()
    app.run()

if __name__ == "__main__":
    main()
