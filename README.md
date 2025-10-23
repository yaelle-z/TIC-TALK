# Multimodal Analysis of Stand-up Comedy: Poses, Topics, Laughter

This repository contains **only the final code** of the multimodal analysis project on stand-up comedy.
Large data files (videos, audio, full `.srt` subtitles), trained models, and heavy intermediate results are hosted separately on a cloud drive.

The notebooks are organized according to the main steps of analysis:
1. **Gestures (poses)** performed by comedians on stage 
2. **Topics** addressed in their performances 
3. **Laughter** triggered in the audience 

---

## 1. Gesture Analysis (Poses)

### `echantillonage_filemp4_toimages.ipynb`
- **Input**: Raw video files (`.mp4`) from the corpus (not included)
- **Output**: A series of images extracted at regular intervals (e.g., 1 frame/second)
- **Purpose**: Generate a representative sample of frames for each show, to be used for annotation and later analysis

### `selection_image_aleatoirement_pour_annoter.ipynb`
- **Input**: Folder containing the extracted video frames
- **Output**: Random subsample of images (CSV of selected files + copies in a dedicated folder)
- **Purpose**: Reduce the number of images to annotate while maintaining diversity in poses and camera shots

### `yolov8_shot_classifier_training.ipynb`
- **Input**: Annotated images labeled with shot types + annotation CSV (exported from Label Studio)
- **Output**: Trained YOLOv8 classification model (`.pt` weights) for shot-type prediction
- **Purpose**: Automate the classification of camera shots (e.g., full-body, medium shot) in extracted video frames


### `image_sorting_by_shot_type(2).ipynb`
- **Input**: Trained YOLOv8 ONNX model + extracted images (organized in subfolders)  
- **Output**: Images copied into folders based on the predicted shot class  
- **Purpose**: Automatically isolate full-body shots and other categories for further gesture analysis


### `image_to_csv_keypoints.py`
- **Input**: Full-body images
- **Output**: One CSV per image with:
  - x, y coordinates of 17 keypoints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles)
  - Bounding box coordinates
- **Purpose**: Convert images into structured numerical data for statistical analysis

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

## 1. Topic Modeling

### `nettoyage_fichiers_srt.ipynb`
- **Input**: Raw subtitle files (`.srt`)
- **Output**: Cleaned `.srt` files (text only, standardized formatting, tags removed)
- **Purpose**: Prepare subtitles for topic modeling (NLP processing)

### `notebook_topic_model_REUSSI.ipynb`
- **Input**: Cleaned `.srt` files
- **Output**:
  - Topic assignment for each time segment
  - Visualizations of topic dynamics
  - CSV export of the results
- **Purpose**: Identify dominant themes in each special using BERTopic and analyze their temporal evolution

---

## 1. Laughter Detection

### `detection_de_rire_whisper_at.py`
- **Input**: Audio or video files (`.mp3`, `.wav`, `.mp4`, `.m4a`)
- **Output**: CSV listing each detected laugh with:
  - Start and end timestamps
  - Laughter type (Whisper-AT classification)
  - Confidence score
- **Purpose**: Automatically detect audience laughter using Whisper-AT



---



