# Multimodal Analysis of Stand-up Comedy: Poses, Topics, Laughter

This repository contains **only the final code** of the multimodal analysis project on stand-up comedy.
Large data files (videos, audio, full `.srt` subtitles), trained models, and heavy intermediate results are hosted separately on a cloud drive.

The notebooks are organized according to the main steps of analysis:
1. **Gestures (poses)** performed by comedians on stage 
2. **Topics** addressed in their performances 
3. **Laughter** triggered in the audience 

---

## 1. Gesture Analysis (Poses)

### `from_video_to_sampled_images.ipynb.ipynb`
- **Input**: Raw video files (`.mp4`) from the corpus (not included)
- **Output**: A series of images extracted at regular intervals (e.g., 1 frame/second)

### `yolov8_shot_classifier_training.ipynb`
- **Input**: Annotated images labeled with shot types + annotation CSV (exported from Label Studio)
- **Output**: Trained YOLOv8 classification model (`.pt` weights) for shot-type prediction
- **Purpose**: Automate the classification of camera shots (e.g., full-body, medium shot) in extracted video frames


### `image_sorting_by_shot_type.ipynb`
- **Input**: Trained YOLOv8 ONNX model + extracted images (organized in subfolders)  
- **Output**: Images copied into folders based on the predicted shot class  
- **Purpose**: Automatically isolate full-body shots and other categories for further gesture analysis


### `image_sample_to_csv_poses.ipynb`
- **Input**: A folder containing `.jpg`, `.png`, or `.jpeg` images of full-body screenshots from stand-up shows.  
- **Output**: One CSV file per image, saved in the output folder, named `image_name-poses_detectees.csv`. Each row contains:  
  - Image name  
  - Bounding box coordinates (`bbox_xmin`, `bbox_ymin`, `bbox_xmax`, `bbox_ymax`)  
  - 2D coordinates (`x`, `y`) for each keypoint, following this order:  
    `Nez`, `Oeil_2`, `Oeil_1`, `Oreille_1`, `Oreille_2`, `Epaule_2`, `Epaule_1`, `Coude_2`, `Coude_1`, `Poignet_2`, `Poignet_1`, `Hanche_2`, `Hanche_1`, `Genou_2`, `Genou_1`, `Cheville_2`, `Cheville_1`
- **Purpose**: Perform pose estimation on a batch of images using `yolov8l-pose.pt` and extract raw, **non-normalized** keypoint coordinates and bounding boxes into separate CSVs for each image.

### `csv_poses_to_cluster.ipynb`
- **Input**: All CSV files of body keypoints
- **Output**:
  - Clusters of similar poses (via clustering)
  - Mean coordinates per cluster
  - Visualizations of average poses
- **Purpose**: Identify recurring postures and describe typical gesture patterns

### `from_annotations_to_trained_model_and_classified_images.ipynb`  
- **Input:** Full-body images + annotation CSV (exported from Label Studio)  
- **Output:**  
  - A trained YOLOv8 classification model (`.pt` file)  
  - A CSV file listing each image and its predicted pose class  

---

## 2. Topic Modeling

### `stand_up_topic_model.ipynb`
- **Input**: Cleaned `.srt` files
- **Output**:
  - Topic assignment for each time segment
  - Visualizations of topic dynamics
  - CSV export of the results
- **Purpose**: Identify dominant themes in each special using BERTopic and analyze their temporal evolution

---

## 3. Laughter Detection

### `laugh_detection_whisper_at.py`
- **Input**: Audio or video files (`.mp3`, `.wav`, `.mp4`, `.m4a`)
- **Output**: One CSV per show listing each detected laugh with:
  - Start and end timestamps
  - Laughter type (Whisper-AT classification)
  - Confidence score
- **Purpose**: Automatically detect audience laughter using Whisper-AT




---



