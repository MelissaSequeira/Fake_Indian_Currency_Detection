from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.widget import Widget
from kivy.core.window import Window

# Set window size (optional)
Window.size = (400, 600)

class CamApp(App):
    def build(self):
        # Main Layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Creating a camera view (85% of the screen)
        camera_layout = BoxLayout(size_hint=(1, 0.85))
        self.camera = Camera(play=True, resolution=(640, 480), allow_stretch=True)  # Camera widget
        camera_layout.add_widget(self.camera)

        # Bottom layout (15% of the screen)
        bottom_layout = BoxLayout(orientation='horizontal', spacing=20, size_hint=(1, 0.15))

        # First half for text labels (recognized currency and detection result)
        text_layout = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.5, 1))

        recognized_currency_label = Label(text="Recognized Currency: None", font_size=14, halign='left', valign='middle')
        detection_result_label = Label(text="Detection: Not Detected", font_size=14, halign='left', valign='middle')

        # Add labels to text layout
        text_layout.add_widget(recognized_currency_label)
        text_layout.add_widget(detection_result_label)

        # Second half for buttons
        button_layout = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.5, 1))

        # Capture Button
        capture_button = Button(text="Capture", font_size=16, size_hint=(1, 0.5), background_normal='', background_color=(0.2, 0.6, 0.9, 1))
        capture_button.bind(on_press=self.capture_image)

        # Recognize Button
        recognize_button = Button(text="Recognize", font_size=16, size_hint=(1, 0.5), background_normal='', background_color=(0.2, 0.9, 0.6, 1))
        recognize_button.bind(on_press=self.recognize_image)

        # Add buttons to button layout
        button_layout.add_widget(capture_button)
        button_layout.add_widget(recognize_button)

        # Add text and button layouts to the bottom layout
        bottom_layout.add_widget(text_layout)
        bottom_layout.add_widget(button_layout)

        # Add both camera and bottom layouts to the main layout
        main_layout.add_widget(camera_layout)
        main_layout.add_widget(bottom_layout)

        return main_layout

    def capture_image(self, instance):
        # Function to capture the image
        print("Capture button clicked!")

    def recognize_image(self, instance):
        # Function to recognize the image
        print("Recognize button clicked!")

if __name__ == '__main__':
    CamApp().run()
