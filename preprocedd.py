import cv2
import os

# Define the base directory for real currency images
base_dir = r"C:\Users\Melissa\AppData\Local\Programs\Python\Python311\fakeDetect\Indian Currency Dataset"  # Update this path

# Define the output dataset directory
output_base_dir = r"C:\Users\Melissa\AppData\Local\Programs\Python\Python311\fakeDetect\preprocessing"  # Where processed images will be saved

# Create the output dataset directories if they don't exist
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Define the resizing factor for resizing images
resize_factor = 0.5  # Adjust this factor as needed

# Define the resizing factor for display
display_resize_factor = 0.25  # Further resize for display

# Define function to resize, save, and manually crop images
def process_image(image_path, denomination):
    # Load the original image
    img = cv2.imread(image_path)
    
    # Resize the image
    height, width = img.shape[:2]
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    resized_img = cv2.resize(img, (new_width, new_height))
    
    # Further resize the image for display
    display_width = int(new_width * display_resize_factor)
    display_height = int(new_height * display_resize_factor)
    display_img = cv2.resize(resized_img, (display_width, display_height))
    
    # Create directories for saving the resized and cropped images
    resized_dir = os.path.join(output_base_dir, denomination, "resized")
    cropped_dir = os.path.join(output_base_dir, denomination, "cropped_features")
    
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
    
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
    
    # Save the resized image
    resized_image_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(resized_dir, resized_image_name), resized_img)
    
    # Show the resized image for manual cropping
    cv2.imshow('Resized Image - Press any key to continue', display_img)
    cv2.waitKey(0)  # Wait for any key to be pressed
    
    # Resize the resized image to fit the screen for cropping
    max_display_size = 800  # Set a maximum size for display window
    if display_img.shape[0] > max_display_size or display_img.shape[1] > max_display_size:
        scale = max_display_size / max(display_img.shape)
        fit_display_img = cv2.resize(display_img, (int(display_img.shape[1] * scale), int(display_img.shape[0] * scale)))
    else:
        fit_display_img = display_img
    
    # Allow the user to select multiple ROIs (regions of interest)
    feature_count = 0
    while True:
        print(f"Select the crop region for {resized_image_name} (drag to select region, press ESC to stop cropping)")
        roi = cv2.selectROI('Crop Image', fit_display_img)  # Use OpenCV's ROI selector
        
        if roi == (0, 0, 0, 0):
            # If no ROI is selected, stop cropping
            print(f"No further cropping region selected for {resized_image_name}.")
            break
        else:
            # Crop the image based on the selected ROI
            cropped_img = resized_img[int(roi[1] / display_resize_factor):int((roi[1] + roi[3]) / display_resize_factor),
                                      int(roi[0] / display_resize_factor):int((roi[0] + roi[2]) / display_resize_factor)]
            
            # Further resize the cropped image for display
            cropped_display_width = int(cropped_img.shape[1] * display_resize_factor)
            cropped_display_height = int(cropped_img.shape[0] * display_resize_factor)
            cropped_display_img = cv2.resize(cropped_img, (cropped_display_width, cropped_display_height))
            
            # Show the cropped image
            cv2.imshow('Cropped Image - Press any key to save', cropped_display_img)
            cv2.waitKey(0)  # Wait for any key to be pressed
            
            # Save the cropped feature image
            cropped_image_name = f"{os.path.splitext(resized_image_name)[0]}_cropped_{feature_count}.jpg"
            cv2.imwrite(os.path.join(cropped_dir, cropped_image_name), cropped_img)
            feature_count += 1
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Process only images from the "500" rupees folder
def process_500_rupees_images(base_dir):
    denomination = "10"  # Process only "500" denomination
    denomination_dir = os.path.join(base_dir, denomination)
    
    if os.path.exists(denomination_dir):
        # Loop through all images in the "500" denomination folder
        for image_file in os.listdir(denomination_dir):
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(denomination_dir, image_file)
                
                # Process and save the resized and cropped images
                process_image(image_path, denomination)
    else:
        print(f"Folder for denomination {denomination} not found.")

# Run the function to process "500" rupees images
process_500_rupees_images(base_dir)

print("Processing complete! All '500' rupees images resized and features saved in the new dataset directory.")
