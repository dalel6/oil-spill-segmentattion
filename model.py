from roboflow import Roboflow
import os 
rf = Roboflow(api_key="Gp7AJzIZsIWWHdlRVAjv")
project = rf.workspace().project("oil-spill-segmentation")
model = project.version(3).model

# Directory where the input images are located
input_directory = "dataset"

# Directory where the segmented images will be saved
output_directory = "datasetsegmentation"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

input_files = [f"img{i}.jpg" for i in range(297, 978)]

# Iterate over each file in the sorted input list
for i, filename in enumerate(input_files):
    input_image_path = os.path.join(input_directory, filename)
    output_image_path = os.path.join(output_directory, f"img_seg{i + 297}.jpg")

  
    # Make predictions and save the annotated image
    model.predict(input_image_path).save(output_image_path)

    print(f"Processed: {input_image_path} -> {output_image_path}")




print("Processing completed.")

