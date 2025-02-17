# import os
# import cv2
# import pandas as pd
# import mediapipe as mp
# import librosa
# import numpy as np
# import subprocess

# # Paths
# video_folder = 'D:/MELD/MELD.Raw/train/train_splits/'
# csv_path = "D:/MELD/MELD.Raw/Csv/train_sent_emo.csv"  # Path to your CSV
# output_face_folder = 'D:/MELD/output_faces/'
# output_audio_folder = 'D:/MELD/output_audio/'

# # Create directories for saving extracted faces and audio if not exist
# if not os.path.exists(output_face_folder):
#     os.makedirs(output_face_folder)

# if not os.path.exists(output_audio_folder):
#     os.makedirs(output_audio_folder)

# # Load CSV
# csv_data = pd.read_csv(csv_path)

# # MediaPipe for face detection and face mesh landmarks
# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh

# # Function to check if the face and audio features already exist
# def check_if_processed(video_name, speaker_name, emotion):
#     # Check if the face frames already exist
#     face_folder = os.path.join(output_face_folder, emotion, speaker_name)
#     audio_folder = os.path.join(output_audio_folder, emotion, speaker_name)
    
#     # Check if the directory exists for face frames
#     if os.path.exists(face_folder):
#         # Check if at least one frame exists in the folder (this is enough to confirm processing)
#         face_files = [f for f in os.listdir(face_folder) if f.endswith('.jpg')]
#         if len(face_files) > 0:
#             return True  # Faces are already processed
    
#     # Check if the audio features already exist
#     if os.path.exists(audio_folder):
#         # Check if at least one audio feature file exists in the folder
#         audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.npy')]
#         if len(audio_files) > 0:
#             return True  # Audio features are already processed

#     return False  # No processed data found

# # Extract face and audio features for all rows
# def extract_faces_and_audio(video_path, speaker_name, emotion, output_face_folder, output_audio_folder):
#     cap = cv2.VideoCapture(video_path)
#     frame_list = []
#     audio_features_list = []

#     # Extract faces and landmarks
#     with mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh:
#         frame_count = 0  # Keep track of the frame number
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"Failed to capture frame in {video_path}. Breaking the loop.")
#                 break

#             # Debugging the frame number
#             frame_count += 1
#             print(f"Processing frame {frame_count} of {video_path}")

#             # Face detection and landmarks extraction
#             results = face_mesh.process(frame)
#             if results.multi_face_landmarks:
#                 for landmarks in results.multi_face_landmarks:
#                     frame_list.append(frame)  # Store the frame with detected face

#     cap.release()

#     # Save extracted faces immediately (instead of after all frames)
#     if frame_list:
#         # Create emotion and speaker-specific folder
#         speaker_face_folder = os.path.join(output_face_folder, emotion, speaker_name)
#         if not os.path.exists(speaker_face_folder):
#             os.makedirs(speaker_face_folder)

#         # Save each frame immediately
#         for idx, face in enumerate(frame_list):
#             # Create a unique filename by adding video name and emotion
#             frame_filename = os.path.join(speaker_face_folder, f'{video_name}_{emotion}_frame_{idx}.jpg')
#             cv2.imwrite(frame_filename, face)
#             print(f"Saved face frame {idx} for {speaker_name} with filename {frame_filename}")

#     # Extract audio features immediately
#     # Extract audio from the video using ffmpeg
#     audio_path = video_path.replace('.mp4', '.wav')  # Output path for the temporary audio file
#     if not os.path.exists(audio_path):  # If the audio file doesn't exist, extract it
#         try:
#             # Use ffmpeg to extract audio from video
#             command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path}"
#             subprocess.run(command, shell=True, check=True)  # Execute the command
#             print(f"Extracted audio for {video_path}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error extracting audio from {video_path}: {e}")
#             return []

#     # Now extract audio features from the newly extracted .wav file
#     if os.path.exists(audio_path):
#         try:
#             y, sr = librosa.load(audio_path)
#             mfcc = librosa.feature.mfcc(y=y, sr=sr)
#             audio_features_list.append(mfcc)  # Store the MFCC features

#             # Save audio features immediately in emotion and speaker folder
#             speaker_audio_folder = os.path.join(output_audio_folder, emotion, speaker_name)
#             if not os.path.exists(speaker_audio_folder):
#                 os.makedirs(speaker_audio_folder)

#             audio_filename = os.path.join(speaker_audio_folder, f'{speaker_name}_audio_features.npy')
#             np.save(audio_filename, mfcc)
#             print(f"Saved audio features for {speaker_name} with emotion {emotion}")
#         except Exception as e:
#             print(f"Error extracting audio features from {audio_path}: {e}")
#     else:
#         print(f"Audio extraction failed for {video_path}")

#     return frame_list, audio_features_list

# # Iterate over CSV and extract features
# for idx, row in csv_data.iterrows():
#     speaker_name = row['Speaker']
#     dialogue_id = row['Dialogue_ID']
#     utterance_id = row['Utterance_ID']
#     emotion = row['Emotion']  # Emotion from the CSV
    
#     # Construct the video name based on Dialogue_ID and Utterance_ID
#     video_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"
#     video_path = os.path.join(video_folder, video_name)
    
#     # Check if the data has already been processed
#     if not check_if_processed(video_name, speaker_name, emotion):
#         # Video exists and has not been processed yet
#         if os.path.exists(video_path):
#             # Extract faces and audio features
#             frames, audio_features = extract_faces_and_audio(video_path, speaker_name, emotion, output_face_folder, output_audio_folder)
#             print(f"Processed {video_name} for speaker {speaker_name} with emotion {emotion}")
#         else:
#             print(f"Video file {video_name} does not exist!")
#     else:
#         print(f"Data for {video_name} for speaker {speaker_name} with emotion {emotion} already processed!")

# print("Data extraction complete!")


















































































































































