import whisper_at
import os
import pandas as pd

def detect_laughs(audio_path):
    # load whisper
    model = whisper_at.load_model('base')
    
    # transcription and ausio labelling
    result = model.transcribe(audio_path, at_time_res=3)  # 3 secondes de résolution
    
    # laugh detection
    audio_tag_result = whisper_at.parse_at_label(result, 
        language='follow_asr', 
        p_threshold=-1,
        include_class_list=[16, 17, 18, 19, 20, 21])  # Laugh type
    
    # prepare data for the dataframe
    laugh_data = []
    for tag in audio_tag_result:
        if tag['audio tags']:
            for laugh_type in tag['audio tags']:
                laugh_data.append({
                    'Start Time (s)': tag['time']['start'],
                    'End Time (s)': tag['time']['end'],
                    'Laugh Type': laugh_type[0],
                    'Confidence': laugh_type[1]
                })
    
    return pd.DataFrame(laugh_data)

def process_comedy_specials(input_folder, output_folder):
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # parse all audio files within the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp3', '.wav', '.mp4', '.m4a')):  
            input_path = os.path.join(input_folder, filename)
            
            try:
                # laugh  detection
                laugh_df = detect_laughs(input_path)
                
                # name of the file output 
                output_filename = os.path.splitext(filename)[0] + '_laughs.csv'
                output_path = os.path.join(output_folder, output_filename)
                
                # save in a csv file
                laugh_df.to_csv(output_path, index=False)
                print(f"Processed {filename}: {len(laugh_df)} laughs detected")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# using
input_folder = 'Captations_mp4/'
output_folder = '/laughs_detect_exp1'
process_comedy_specials(input_folder, output_folder)
