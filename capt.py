import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture

class CurrencyRecognitionApp(App):
    def build(self):
        # Load the trained model (.h5)
        self.model = tf.keras.models.load_model('currency_recognition_model.h5', compile=False)

        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Camera layout (85% of the screen)
        camera_layout = BoxLayout(size_hint=(1, 0.85))
        self.camera = Camera(play=True, resolution=(640, 480))
        camera_layout.add_widget(self.camera)

        # Bottom layout (15% of the screen)
        bottom_layout = BoxLayout(orientation='horizontal', spacing=20, size_hint=(1, 0.15))

        # Label to display result
        self.result_label = Label(text="Currency Status: Not Detected", font_size=14, halign='center', valign='middle')

        # Button to capture and recognize
        capture_button = Button(text="Capture & Recognize", font_size=16, background_normal='', background_color=(0.2, 0.6, 0.9, 1))
        capture_button.bind(on_press=self.capture_and_recognize)

        # Add result label and button to the bottom layout
        bottom_layout.add_widget(self.result_label)
        bottom_layout.add_widget(capture_button)

        # Add the camera layout and bottom layout to the main layout
        main_layout.add_widget(camera_layout)
        main_layout.add_widget(bottom_layout)

        return main_layout

    def capture_and_recognize(self, instance):
        # Capture the current frame from the camera
        texture = self.camera.texture
        if texture:
            img_data = np.frombuffer(texture.pixels, np.uint8)
            img_data = img_data.reshape(texture.height, texture.width, 4)  # RGBA format
            img_data = img_data[:, :, :3]  # Convert to RGB

            # Convert captured image to grayscale for processing
            gray_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)

            # Use edge detection to find the currency
            edges = cv2.Canny(gray_img, 50, 150)  # Adjust these parameters as needed

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours for debugging
            debug_img = img_data.copy()
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            cv2.imshow("Contours", debug_img)  # Display for debugging

            # If contours are found, crop and process the image
            if contours:
                # Get the largest contour by area
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Crop the image based on detected contour
                cropped_image = img_data[y:y+h, x:x+w]

                # Resize the cropped image to the required input size for the model
                resized_image = cv2.resize(cropped_image, (128, 128))  # Ensure it's 128x128

                # Normalize the image as per model requirements
                normalized_image = resized_image / 255.0

                # Expand dimensions to match the model input (batch size 1, 128, 128, 3)
                input_image = np.expand_dims(normalized_image, axis=0)

                # Predict using the model
                prediction = self.model.predict(input_image)

                # Example: Assuming the model returns a classification
                recognized_class = np.argmax(prediction)  # Get the predicted class

                # Map the recognized class to currency denominations
                denomination_map = {
                    0: '10',
                    1: '20',
                    2: '50',
                    3: '100',
                    4: '200',
                    5: '500'
                }

                currency_denomination = denomination_map.get(recognized_class, 'Unknown')

                # Update the UI with the recognized currency result
                self.result_label.text = f"Detected Currency: {currency_denomination}"
            else:
                self.result_label.text = "No Currency Detected"

if __name__ == '__main__':
    CurrencyRecognitionApp().run()
