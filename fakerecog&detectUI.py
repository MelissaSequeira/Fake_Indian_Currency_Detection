import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView

# Set window size (optional)
Window.size = (400, 600)

class FeatureDisplayPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = 'Detected Features'
        self.size_hint = (0.8, 0.8)
        self.content = BoxLayout(orientation='vertical')
        
        # Create a ScrollView to handle large content
        scroll_view = ScrollView(size_hint=(1, 1))
        self.feature_label = Label(size_hint_y=None, text='', halign='left', valign='top', padding=[10, 10])
        self.feature_label.bind(width=self.update_width)
        scroll_view.add_widget(self.feature_label)
        
        self.content.add_widget(scroll_view)

    def update_content(self, features_text):
        self.feature_label.text = features_text
        self.feature_label.texture_update()
        self.feature_label.height = self.feature_label.texture_size[1]  # Update the height based on text

    def update_width(self, instance, value):
        self.feature_label.text_size = (value - 20, None)  # Adjust text size to fit within popup width

class CurrencyRecognitionApp(App):
    def build(self):
        self.main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Creating a camera view (85% of the screen)
        camera_layout = BoxLayout(size_hint=(1, 0.85))
        self.camera = Camera(play=True, resolution=(640, 480), allow_stretch=True)  # Camera widget
        camera_layout.add_widget(self.camera)

        # Bottom layout (15% of the screen)
        bottom_layout = BoxLayout(orientation='horizontal', spacing=20, size_hint=(1, 0.15))

        # First half for text labels (recognized currency and detection result)
        text_layout = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.5, 1))
        self.recognized_currency_label = Label(text="Recognized Currency: None", font_size=14, halign='left', valign='middle')
        self.detection_result_label = Label(text="Detection: Not Detected", font_size=14, halign='left', valign='middle')
        text_layout.add_widget(self.recognized_currency_label)
        text_layout.add_widget(self.detection_result_label)

        # Second half for buttons
        button_layout = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.5, 1))
        self.capture_button = Button(text="Capture", font_size=16, size_hint=(1, 0.5), background_normal='', background_color=(0.2, 0.6, 0.9, 1))
        self.capture_button.bind(on_press=self.capture_image)
        button_layout.add_widget(self.capture_button)

        # Add text and button layouts to the bottom layout
        bottom_layout.add_widget(text_layout)
        bottom_layout.add_widget(button_layout)

        # Add both camera and bottom layouts to the main layout
        self.main_layout.add_widget(camera_layout)
        self.main_layout.add_widget(bottom_layout)

        # Load the trained classification model for denomination recognition
        self.denomination_model = tf.keras.models.load_model('currency_recognition_model.h5', compile=False)

        # Load the trained model for feature detection
        self.feature_model = tf.keras.models.load_model('currency_detection_modelnew.h5')

        # The labels corresponding to the trained classes (folder names)
        self.feature_labels = [
            '100_backpic', '100_backpicname', '100_bleed_lines', '100_bubbles', '100_devfig', '100_devnum',
            '100_engfig', '100_engnum', '100_flig', '100_registrate', '100_seminum', '100_serialno',
            '100_uv_lines', '100_vi_code', '100_year', '10_10devfig', '10_10flig', '10_devnum', '10_engnum',
            '10_n_backpic', '10_n_backpicname', '10_n_serialno', '10_n_year', '10_o_10corner', '10_o_10front',
            '10_o_backpic', '10_o_serialno', '200_backpic', '200_backpicname', '200_bleed_lines', '200_bubbles',
            '200_devfig', '200_devnum', '200_engfig', '200_engnum', '200_green', '200_registrate', '200_serialno',
            '200_uv_lines', '200_vi_code', '200_year', '20_flig', '20_n_backpic',
            '20_n_backpicname', '20_n_devfig', '20_n_devnum', '20_n_engfig', '20_n_engnum', '20_n_serialno',
            '20_n_year', '20_o_engdev', '20_o_registrate', '20_o_serialno', '20_seminum', '500_backpicname',
            '500_bleed_lines', '500_bubbles', '500_devfig', '500_devnum', '500_engfig', '500_engnum',
            '500_flig', '500_green', '500_registrate', '500_serialno', '500_uv_lines', '500_vi_code',
            '500_year', '50_backpic', '50_backpicname', '50_bubbles', '50_deepnum', '50_devfig', '50_devnum',
            '50_engfig', '50_engnum', '50_flig', '50_flig2', '50_serialno', '50_year', 'difflang', 'gandhiji',
            'logo', 'nationalamblem', 'rbi_dev', 'rbi_eng', 'swatchbharat'
        ]

        self.currency_denomination_map = {
            0: '10',
            1: '20',
            2: '50',
            3: '100',
            4: '200',
            5: '500'
        }

        self.captured_frame = None
        self.recognized_denomination = None

        self.feature_display_popup = FeatureDisplayPopup()  # Initialize the popup

        return self.main_layout

    def capture_image(self, instance):
        texture = self.camera.texture
        if texture:
            img_data = np.frombuffer(texture.pixels, np.uint8)
            img_data = img_data.reshape(texture.height, texture.width, 4)  # RGBA format
            img_data = img_data[:, :, :3]  # Convert to RGB

            self.captured_frame = img_data.copy()  # Ensure we have a writable copy
            self.recognize_denomination(self.captured_frame)

    def recognize_denomination(self, frame):
        gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_img, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_denomination = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = frame[y:y+h, x:x+w]
            resized_image = cv2.resize(cropped_image, (128, 128))
            normalized_image = resized_image / 255.0
            input_image = np.expand_dims(normalized_image, axis=0)

            prediction = self.denomination_model.predict(input_image)
            recognized_class = np.argmax(prediction)
            detected_denomination = self.currency_denomination_map.get(recognized_class, 'Unknown')

            if detected_denomination != 'Unknown':
                self.recognized_denomination = detected_denomination
                break

        if detected_denomination:
            self.recognized_currency_label.text = f"Recognized Currency: {detected_denomination}"
            self.process_features()
        else:
            self.recognized_currency_label.text = "Recognized Currency: None"
            self.detection_result_label.text = "Detection: Not Detected"

    def process_features(self):
        if self.recognized_denomination:
            detected_features_text = ""
            feature_labels_to_check = [label for label in self.feature_labels if label.startswith(f"{self.recognized_denomination}_")]

            if feature_labels_to_check:
                detected_features_text = "\n".join(feature_labels_to_check)
            else:
                detected_features_text = "No features detected"

            self.feature_display_popup.update_content(detected_features_text)  # Update and show the popup
            self.feature_display_popup.open()

if __name__ == '__main__':
    CurrencyRecognitionApp().run()
