# Grad-CAM Visualization for Machine Learning Models

This project demonstrates how to visualize Grad-CAM heatmaps for convolutional neural networks (CNNs). Grad-CAM is used to interpret and visualize which regions of an input image contribute most to the model's predictions.

## Features

- Load a pre-trained deep learning model.
- Generate Grad-CAM heatmaps for a specified convolutional layer.
- Overlay Grad-CAM heatmaps on original images for better interpretability.
- Display Grad-CAM results in a grid for multiple images.

## Requirements

- Python 3.7 or above
- TensorFlow 2.x
- NumPy
- Matplotlib

## How to Use
 Load your pre-trained model: Update the path_224 variable with the path to your model file.
 
 Test the model: Run the script to confirm the model processes dummy inputs successfully.
 
 Visualize Grad-CAM:
- Specify the target convolutional layer in the last_conv_layer_name variable.
- Provide the folder containing the test images in base_folder.
- Adjust grid size using the grid_size parameter.

## Output
Grad-CAM heatmaps overlayed on input images will be displayed in a grid.
The title above each image indicates the predicted class.

## Example
Example visualization of Grad-CAM:
![image](https://github.com/user-attachments/assets/c51664fd-d6ae-4d58-96d6-4cddc899b713)

## Author
Kai-Yu Tseng

Install the dependencies using pip:



```bash
pip install tensorflow numpy matplotlib
