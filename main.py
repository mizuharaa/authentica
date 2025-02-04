import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.label import Label

# Enhanced Deep Fake Detection Model for Images with more complex architecture
def build_image_model():
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(512, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

# Enhanced Deep Fake Detection Model for Videos with temporal features
def build_video_model():
    model = Sequential([
        Conv2D(128, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(512, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(512, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

class DeepFakeDetectorApp(App):
    def build(self):
        self.image_model = build_image_model()
        self.video_model = build_video_model()
        
        # Load pre-trained weights
        try:
            self.image_model.load_weights('image_model_weights.h5')
            self.video_model.load_weights('video_model_weights.h5')
        except:
            print("Warning: Pre-trained weights not found. Models will give random predictions.")
        
        self.is_recording = False
        self.video_frames = []
        
        # Main Layout
        layout = BoxLayout(orientation='vertical')
        
        # Camera preview
        self.camera = Camera(play=True, resolution=(640, 480))
        layout.add_widget(self.camera)
        
        # Buttons
        button_layout = BoxLayout(size_hint_y=0.2)
        
        capture_btn = Button(text='Capture Image')
        capture_btn.bind(on_press=self.capture_image)
        
        self.video_btn = Button(text='Start Recording')
        self.video_btn.bind(on_press=self.toggle_video_recording)
        
        upload_btn = Button(text='Upload Media')
        upload_btn.bind(on_press=self.show_file_chooser)
        
        button_layout.add_widget(capture_btn)
        button_layout.add_widget(self.video_btn)
        button_layout.add_widget(upload_btn)
        
        layout.add_widget(button_layout)
        
        # Result label
        self.result_label = Label(size_hint_y=0.1)
        layout.add_widget(self.result_label)
        
        return layout
    
    def preprocess_image(self, image):
        # Enhanced preprocessing
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0
        
        # Apply additional preprocessing
        image = tf.image.per_image_standardization(image)
        return np.expand_dims(image, axis=0)
    
    def detect_deepfake_image(self, image):
        processed_image = self.preprocess_image(image)
        
        # Multiple predictions with test-time augmentation
        predictions = []
        
        # Original image
        pred = self.image_model.predict(processed_image)
        predictions.append(pred[0][0])
        
        # Flipped image
        flipped = tf.image.flip_left_right(processed_image)
        pred = self.image_model.predict(flipped)
        predictions.append(pred[0][0])
        
        # Slightly rotated images
        for angle in [-15, 15]:
            rotated = tf.image.rot90(processed_image, k=angle//90)
            pred = self.image_model.predict(rotated)
            predictions.append(pred[0][0])
        
        # Average predictions
        final_prediction = np.mean(predictions)
        
        # Apply confidence threshold
        confidence = abs(final_prediction - 0.5) * 2  # Scale to 0-1
        return final_prediction, confidence
    
    def detect_deepfake_video(self, video_frames):
        if not video_frames:
            return 0.5, 0.0
        
        predictions = []
        confidences = []
        
        for frame in video_frames:
            pred, conf = self.detect_deepfake_image(frame)
            predictions.append(pred)
            confidences.append(conf)
        
        # Temporal smoothing
        window_size = 5
        smoothed_predictions = np.convolve(predictions, 
                                         np.ones(window_size)/window_size, 
                                         mode='valid')
        
        final_prediction = np.mean(smoothed_predictions)
        final_confidence = np.mean(confidences)
        
        return final_prediction, final_confidence
    
    def show_result(self, prediction, confidence):
        if confidence < 0.3:
            result = f"Uncertain ({prediction:.1%} chance of deepfake, low confidence)"
        else:
            if prediction > 0.7:
                result = f"LIKELY DEEPFAKE ({prediction:.1%} probability)"
            elif prediction < 0.3:
                result = f"LIKELY REAL ({(1-prediction):.1%} probability)"
            else:
                result = f"UNCERTAIN ({prediction:.1%} chance of deepfake)"
        
        self.result_label.text = result
    
    def capture_image(self, instance):
        texture = self.camera.texture
        size = texture.size
        pixels = texture.pixels
        
        image = np.frombuffer(pixels, dtype=np.uint8)
        image = image.reshape(size[1], size[0], 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        prediction, confidence = self.detect_deepfake_image(image)
        self.show_result(prediction, confidence)
    
    def toggle_video_recording(self, instance):
        if not self.is_recording:
            self.is_recording = True
            self.video_btn.text = 'Stop Recording'
            self.video_frames = []
            Clock.schedule_interval(self.record_frame, 1.0/30.0)  # 30 FPS
        else:
            self.is_recording = False
            self.video_btn.text = 'Start Recording'
            Clock.unschedule(self.record_frame)
            
            # Process recorded video
            prediction, confidence = self.detect_deepfake_video(self.video_frames)
            self.show_result(prediction, confidence)
    
    def record_frame(self, dt):
        texture = self.camera.texture
        size = texture.size
        pixels = texture.pixels
        
        frame = np.frombuffer(pixels, dtype=np.uint8)
        frame = frame.reshape(size[1], size[0], 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        self.video_frames.append(frame)
    
    def show_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserListView(
            filters=['*.jpg', '*.jpeg', '*.png', '*.mp4', '*.avi'])
        
        content.add_widget(file_chooser)
        
        # Buttons
        button_layout = BoxLayout(size_hint_y=0.2)
        
        select_btn = Button(text='Select')
        cancel_btn = Button(text='Cancel')
        
        button_layout.add_widget(select_btn)
        button_layout.add_widget(cancel_btn)
        content.add_widget(button_layout)
        
        popup = Popup(title='Choose Media File',
                     content=content,
                     size_hint=(0.9, 0.9))
        
        def select(instance):
            if file_chooser.selection:
                self.process_uploaded_file(file_chooser.selection[0])
                popup.dismiss()
        
        select_btn.bind(on_press=select)
        cancel_btn.bind(on_press=popup.dismiss)
        
        popup.open()
    
    def process_uploaded_file(self, filepath):
        if filepath.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(filepath)
            if image is not None:
                prediction, confidence = self.detect_deepfake_image(image)
                self.show_result(prediction, confidence)
            else:
                self.result_label.text = "Error: Could not load image"
        
        elif filepath.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(filepath)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if frames:
                prediction, confidence = self.detect_deepfake_video(frames)
                self.show_result(prediction, confidence)
            else:
                self.result_label.text = "Error: Could not load video"

if __name__ == '__main__':
    DeepFakeDetectorApp().run()

