import pandas as pd
import os
import librosa

# Step 1: Load the CSV file into a DataFrame
csv_file_path = "C:/Users/alvar/PycharmProjects/corilga_api/dataset/global_data.csv"
df = pd.read_csv(csv_file_path)
'''
# Step 2: Define a function to check if an audio file is valid (not 0 length)
def is_valid_audio(audio_path):
    try:
        audio, _ = librosa.load(audio_path, sr=None)
        # Check if audio length is greater than 0
        return len(audio) > 0
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False

# Step 3: Filter the DataFrame to keep only rows with valid audio
df = df[df['path'].apply(is_valid_audio)]

# Step 4: Save the updated DataFrame back to a CSV file
df.to_csv(csv_file_path, index=False)
'''
num_rows_updated = len(df)
print(f"Rows with audio length 0 removed. Updated CSV saved to {csv_file_path}")
print(f"Number of rows in the updated CSV: {num_rows_updated}")
