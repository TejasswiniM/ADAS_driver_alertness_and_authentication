import os
import cv2
import albumentations as A

# current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the input and output directories
input_dir = os.path.join(current_directory, "..", "face_data", "vid_3", "selected_faces")
output_dir = os.path.join(current_directory, "..", "face_data", "vid_3", "aug_faces", "Teju")

# Define a list of augmentation transformations
transforms = [
   A.Rotate(limit=60),
   A.HorizontalFlip(p=1),     # Horizontal flip with a probability of 0.5
   #A.RandomRotate90(p=1),     # Randomly rotate by 90 degrees with a probability of 0.5
   A.RandomBrightnessContrast(p=0.5),  # Randomly adjust brightness and contrast
   A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
   A.GaussianBlur (p=0.3),
]

# List all files in the input directory
file_list = os.listdir(input_dir)

# Apply different augmentations to each image and save in the output directory
for file_name in file_list:
   if file_name.endswith(('.jpg', '.png', '.jpeg')):
       # Load the image using OpenCV
       image = cv2.imread(os.path.join(input_dir, file_name))
       
       base_name, ext = os.path.splitext(file_name)
       original_img_path = os.path.join(output_dir, f"{base_name}{ext}")
       cv2.imwrite(original_img_path, image)
       augment_img = image.copy()
       
       # Apply a random transformation from the list
       for transform in transforms:
           augmented = transform(image=augment_img)
           augmented_image = augmented["image"]
                     
           # Generate a unique filename for the augmented image
           base_name, ext = os.path.splitext(file_name)
           output_file = os.path.join(output_dir, f"{base_name}_{transform.__class__.__name__}{ext}")

           # Save the augmented image
           cv2.imwrite(output_file, augmented_image)
           #cv2.imshow("img", augmented_image)
           #cv2.waitKey(0)
           
print("Data augmentation and saving completed.")
#cv2.destroyAllWindows()
