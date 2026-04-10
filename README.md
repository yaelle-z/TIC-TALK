# Multimodal Analysis of Stand-up Comedy: Poses, Topics, Laughter


The notebooks and pythons files are organized by modality : 
1. **Image**
2. **Text** 
3. **Sound**

---

## 1. Image
### `from_video_to_sampled_images.ipynb.ipynb`
- **Input**: Raw video files (`.mp4`) from the corpus (not included)
- **Output**: A series of images extracted at regular intervals (e.g., 1 frame/second)

### `yolov8_shot_classifier_training.ipynb`
- **Input**: Annotated images labeled with shot types + annotation CSV (exported from Label Studio)
- **Output**: Trained YOLOv8 classification model (`.pt` weights) for shot-type prediction
- **Purpose**: Automate the classification of camera shots (e.g., full-body, medium shot) in extracted video frames


### `application_shot_classifier_from_model_to_csv_per_show.py`







---

## 2. Text

### `stand_up_topic_model.ipynb`
- **Input**: Cleaned `.srt` files
- **Output**:
  - Topic assignment for each time segment
  - Visualizations of topic dynamics
  - CSV export of the results
- **Purpose**: Identify dominant themes in each special using BERTopic and analyze their temporal evolution

---

## 3. Sound

### `laugh_detection_whisper_at.py`
- **Input**: Audio or video files (`.mp3`, `.wav`, `.mp4`, `.m4a`)
- **Output**: One CSV per show listing each detected laugh with:
  - Start and end timestamps
  - Laughter type (Whisper-AT classification)
  - Confidence score
- **Purpose**: Automatically detect audience laughter using Whisper-AT




---



