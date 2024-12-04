# License Plate Detection

This project is a machine learning-based application designed to detect license plates in images and videos using the YOLOv8 model. It consists of two main components:
1. **Model Training**: A Jupyter Notebook for training a YOLOv8 model on a custom dataset of license plates.
2. **Streamlit Web Application**: A user-friendly web interface for real-time detection of license plates in uploaded media files.

---

## Live Application

The application is deployed on **Streamlit Community Cloud**. You can access it using the following link:

[License Plate Detection Web App](https://licenseplatedetection-vslwwyk2qmgennf8oqpcz4.streamlit.app/)

---

## Project Components

### 1. Model Training (`cars-license-plate-yolov8.ipynb`)
This Jupyter Notebook trains the YOLOv8 model to detect license plates. The workflow includes:
- **Data Preparation**: Prepares a dataset of images with annotated license plates.
- **Model Training**: Fine-tunes a pre-trained YOLOv8 model on the dataset.
- **Validation and Testing**: Evaluates the model's performance on unseen data.
- **Model Export**: Saves the trained model weights (`best_license_plate_model.pt`) for inference.

### 2. Streamlit Web Application (`yolo_application.py`)
This Python script creates a Streamlit web application that allows users to:
- Upload images or videos.
- Process uploaded media using the trained YOLOv8 model.
- Display the results with bounding boxes around detected license plates.

---

## Features

- **Image and Video Support**: Detects license plates in images (JPG, PNG, etc.) and videos (MP4, AVI, etc.).
- **User-Friendly Interface**: Streamlit provides an intuitive platform for easy interaction with the app.
- **Real-Time Processing**: Processes media and displays results with minimal delay.
- **Scalability**: Can be extended for deployment in real-world applications.

---

## Requirements

Install the required dependencies using the following command:
```bash
pip install streamlit ultralytics opencv-python pandas numpy matplotlib
```

---

# Running the Application

## 1. Model Training
1. Open `cars-license-plate-yolov8.ipynb` in a Jupyter environment.
2. Train the YOLOv8 model using your dataset.
3. Save the trained model as `best_license_plate_model.pt`.

## 2. Streamlit Web Application
1. Place the trained model file (`best_license_plate_model.pt`) in the application directory.
2. Run the application using the following command:
   ```bash
   streamlit run yolo_application.py
   ```
3. Open the app in your browser (typically at [http://localhost:8501](http://localhost:8501)).
4. Upload an image or video through the interface.
5. View the detection results with bounding boxes drawn around the detected license plates.

---

## Key Functions in `yolo_application.py`

### 1. `predict_and_save_image(path_test_car, output_image_path)`
- **Purpose**: Processes an uploaded image and detects license plates.
- **Functionality**: Saves the processed image with bounding boxes drawn around detected plates.

### 2. `predict_and_plot_video(video_path, output_path)`
- **Purpose**: Processes a video file frame-by-frame to detect license plates.
- **Functionality**: Saves the processed video with bounding boxes.

### 3. `process_media(input_path, output_path)`
- **Purpose**: Determines whether the uploaded file is an image or video.
- **Functionality**: Routes the file to the appropriate function (`predict_and_save_image` for images or `predict_and_plot_video` for videos) for processing.

---

## Results

- The trained YOLOv8 model achieves high accuracy in detecting license plates.
- The Streamlit app provides a seamless real-time interface for processing and visualizing license plate detection in both images and videos.

---

## Conclusion

This application demonstrates the effective use of the YOLOv8 model for real-time license plate detection. By combining a powerful detection model with a user-friendly web interface, this solution is well-suited for practical applications such as:
- Traffic monitoring
- Automated toll collection
- Vehicle tracking systems

This project showcases how advanced machine learning models can be made accessible for real-world deployment.