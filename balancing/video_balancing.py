# # import os
# # import cv2
# # import pandas as pd
# # import mediapipe as mp
# # import librosa
# # import numpy as np
# # import subprocess

# # # Paths
# # video_folder = 'D:/MELD/MELD.Raw/train/train_splits/'
# # csv_path = "D:/MELD/MELD.Raw/Csv/train_sent_emo.csv"  # Path to your CSV
# # output_face_folder = 'D:/MELD/output_faces/'
# # output_audio_folder = 'D:/MELD/output_audio/'

# # # Create directories for saving extracted faces and audio if not exist
# # if not os.path.exists(output_face_folder):
# #     os.makedirs(output_face_folder)

# # if not os.path.exists(output_audio_folder):
# #     os.makedirs(output_audio_folder)

# # # Load CSV
# # csv_data = pd.read_csv(csv_path)

# # # MediaPipe for face detection and face mesh landmarks
# # mp_face_detection = mp.solutions.face_detection
# # mp_face_mesh = mp.solutions.face_mesh

# # # Function to check if the face and audio features already exist
# # def check_if_processed(video_name, speaker_name, emotion):
# #     # Check if the face frames already exist
# #     face_folder = os.path.join(output_face_folder, emotion, speaker_name)
# #     audio_folder = os.path.join(output_audio_folder, emotion, speaker_name)
    
# #     # Check if the directory exists for face frames
# #     if os.path.exists(face_folder):
# #         # Check if at least one frame exists in the folder (this is enough to confirm processing)
# #         face_files = [f for f in os.listdir(face_folder) if f.endswith('.jpg')]
# #         if len(face_files) > 0:
# #             return True  # Faces are already processed
    
# #     # Check if the audio features already exist
# #     if os.path.exists(audio_folder):
# #         # Check if at least one audio feature file exists in the folder
# #         audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.npy')]
# #         if len(audio_files) > 0:
# #             return True  # Audio features are already processed

# #     return False  # No processed data found

# # # Extract face and audio features for all rows
# # def extract_faces_and_audio(video_path, speaker_name, emotion, output_face_folder, output_audio_folder):
# #     cap = cv2.VideoCapture(video_path)
# #     frame_list = []
# #     audio_features_list = []

# #     # Extract faces and landmarks
# #     with mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh:
# #         frame_count = 0  # Keep track of the frame number
        
# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 print(f"Failed to capture frame in {video_path}. Breaking the loop.")
# #                 break

# #             # Debugging the frame number
# #             frame_count += 1
# #             print(f"Processing frame {frame_count} of {video_path}")

# #             # Face detection and landmarks extraction
# #             results = face_mesh.process(frame)
# #             if results.multi_face_landmarks:
# #                 for landmarks in results.multi_face_landmarks:
# #                     frame_list.append(frame)  # Store the frame with detected face

# #     cap.release()

# #     # Save extracted faces immediately (instead of after all frames)
# #     if frame_list:
# #         # Create emotion and speaker-specific folder
# #         speaker_face_folder = os.path.join(output_face_folder, emotion, speaker_name)
# #         if not os.path.exists(speaker_face_folder):
# #             os.makedirs(speaker_face_folder)

# #         # Save each frame immediately
# #         for idx, face in enumerate(frame_list):
# #             # Create a unique filename by adding video name and emotion
# #             frame_filename = os.path.join(speaker_face_folder, f'{video_name}_{emotion}_frame_{idx}.jpg')
# #             cv2.imwrite(frame_filename, face)
# #             print(f"Saved face frame {idx} for {speaker_name} with filename {frame_filename}")

# #     # Extract audio features immediately
# #     # Extract audio from the video using ffmpeg
# #     audio_path = video_path.replace('.mp4', '.wav')  # Output path for the temporary audio file
# #     if not os.path.exists(audio_path):  # If the audio file doesn't exist, extract it
# #         try:
# #             # Use ffmpeg to extract audio from video
# #             command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path}"
# #             subprocess.run(command, shell=True, check=True)  # Execute the command
# #             print(f"Extracted audio for {video_path}")
# #         except subprocess.CalledProcessError as e:
# #             print(f"Error extracting audio from {video_path}: {e}")
# #             return []

# #     # Now extract audio features from the newly extracted .wav file
# #     if os.path.exists(audio_path):
# #         try:
# #             y, sr = librosa.load(audio_path)
# #             mfcc = librosa.feature.mfcc(y=y, sr=sr)
# #             audio_features_list.append(mfcc)  # Store the MFCC features

# #             # Save audio features immediately in emotion and speaker folder
# #             speaker_audio_folder = os.path.join(output_audio_folder, emotion, speaker_name)
# #             if not os.path.exists(speaker_audio_folder):
# #                 os.makedirs(speaker_audio_folder)

# #             audio_filename = os.path.join(speaker_audio_folder, f'{speaker_name}_audio_features.npy')
# #             np.save(audio_filename, mfcc)
# #             print(f"Saved audio features for {speaker_name} with emotion {emotion}")
# #         except Exception as e:
# #             print(f"Error extracting audio features from {audio_path}: {e}")
# #     else:
# #         print(f"Audio extraction failed for {video_path}")

# #     return frame_list, audio_features_list

# # # Iterate over CSV and extract features
# # for idx, row in csv_data.iterrows():
# #     speaker_name = row['Speaker']
# #     dialogue_id = row['Dialogue_ID']
# #     utterance_id = row['Utterance_ID']
# #     emotion = row['Emotion']  # Emotion from the CSV
    
# #     # Construct the video name based on Dialogue_ID and Utterance_ID
# #     video_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"
# #     video_path = os.path.join(video_folder, video_name)
    
# #     # Check if the data has already been processed
# #     if not check_if_processed(video_name, speaker_name, emotion):
# #         # Video exists and has not been processed yet
# #         if os.path.exists(video_path):
# #             # Extract faces and audio features
# #             frames, audio_features = extract_faces_and_audio(video_path, speaker_name, emotion, output_face_folder, output_audio_folder)
# #             print(f"Processed {video_name} for speaker {speaker_name} with emotion {emotion}")
# #         else:
# #             print(f"Video file {video_name} does not exist!")
# #     else:
# #         print(f"Data for {video_name} for speaker {speaker_name} with emotion {emotion} already processed!")

# # print("Data extraction complete!")












































































# # import tensorflow as tf
# # import os
# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt
# # from keras.api import layers, models, optimizers
# # from keras.api.callbacks import ModelCheckpoint
# # from keras.api.losses import BinaryCrossentropy
# # # Check for GPU
# # def check_gpu():
# #     devices = tf.config.list_physical_devices('GPU')
# #     if devices:
# #         print(f"TensorFlow is using the following GPU: {devices[0].name}")
# #     else:
# #         print("TensorFlow is not using any GPU.")

# # check_gpu()

# # # Directories
# # face_data_dir = 'D:/MELD/output_faces/'  # Faces data folder
# # audio_data_dir = 'D:/MELD/output_audio/'  # Audio features folder

# # # Function to check directory structure and print the structure
# # def check_directory_structure():
# #     print("Checking directory structure...")
# #     if not os.path.exists(face_data_dir):
# #         print(f"Error: {face_data_dir} does not exist.")
# #         return False
# #     if not os.path.exists(audio_data_dir):
# #         print(f"Error: {audio_data_dir} does not exist.")
# #         return False

# #     # Print the folder structure for faces and audio directories
# #     print("Face data directory structure:")
# #     for root, dirs, files in os.walk(face_data_dir):
# #         level = root.replace(face_data_dir, '').count(os.sep)
# #         indent = ' ' * 4 * level
# #         print(f"{indent}{os.path.basename(root)}/")
# #         for f in files:
# #             print(f"{indent}    {f}")

# #     print("Audio data directory structure:")
# #     for root, dirs, files in os.walk(audio_data_dir):
# #         level = root.replace(audio_data_dir, '').count(os.sep)
# #         indent = ' ' * 4 * level
# #         print(f"{indent}{os.path.basename(root)}/")
# #         for f in files:
# #             print(f"{indent}    {f}")
    
# #     # Check if emotion folders exist in both face and audio directories
# #     emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
# #     for emotion in emotions:
# #         if not os.path.exists(os.path.join(face_data_dir, emotion)):
# #             print(f"Error: Emotion folder '{emotion}' not found in {face_data_dir}.")
# #             return False
# #         if not os.path.exists(os.path.join(audio_data_dir, emotion)):
# #             print(f"Error: Emotion folder '{emotion}' not found in {audio_data_dir}.")
# #             return False

# #     print("Directory structure is valid.")
# #     return True

# # # Load all images and audio for training
# # def load_data(emotions=None, speakers=None):
# #     if emotions is None:
# #         emotions = os.listdir(face_data_dir)  # Automatically detect available emotions
# #     if speakers is None:
# #         speakers = []  # List to hold speakers

# #     data = {}
# #     for emotion in emotions:
# #         if emotion not in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']:
# #             continue
# #         data[emotion] = {}
# #         emotion_folder = os.path.join(face_data_dir, emotion)
        
# #         for speaker in os.listdir(emotion_folder):
# #             speaker_folder = os.path.join(emotion_folder, speaker)
# #             if os.path.isdir(speaker_folder):
# #                 print(f"Loading data for {emotion} emotion and speaker {speaker}...")
# #                 data[emotion][speaker] = {
# #                     'faces': load_images(emotion, speaker),
# #                     'audio': load_audio_features(emotion, speaker)
# #                 }
# #     return data

# # # Function to load face images for a specific emotion and speaker
# # def load_images(emotion, speaker, image_size=(256, 256)):
# #     emotion_folder = os.path.join(face_data_dir, emotion, speaker)
# #     images = []
# #     for filename in os.listdir(emotion_folder):
# #         if filename.endswith('.jpg'):
# #             img_path = os.path.join(emotion_folder, filename)
# #             img = cv2.imread(img_path)
# #             img = cv2.resize(img, image_size)
# #             images.append(img)
# #     return np.array(images)

# # # Function to load audio features for a specific emotion and speaker
# # def load_audio_features(emotion, speaker):
# #     emotion_folder = os.path.join(audio_data_dir, emotion, speaker)
# #     audio_features = []
# #     for filename in os.listdir(emotion_folder):
# #         if filename.endswith('.npy'):
# #             audio_path = os.path.join(emotion_folder, filename)
# #             feature = np.load(audio_path)
# #             audio_features.append(feature)
# #     return np.array(audio_features)

# # # CycleGAN Model Setup
# # import tensorflow as tf
# # import numpy as np

# # # Define your "patience" for early stopping
# # early_stopping_patience = 20
# # min_generator_loss = np.inf  # Initialize with a high value to track improvement
# # best_generator_loss_epoch = 0

# # # Keep track of the number of epochs with no improvement
# # epochs_without_improvement = 0

# # # Save checkpoints at regular intervals
# # checkpoint_dir = "./checkpoints"
# # checkpoint_prefix = checkpoint_dir + "/checkpoint.weights.h5"  # Add .weights.h5 extension

# # # Define your checkpoint callback
# # checkpoint_callback = ModelCheckpoint(
# #     checkpoint_prefix, 
# #     save_weights_only=True, 
# #     save_best_only=True, 
# #     monitor="total_g_loss",
# #     verbose=1
# # )

# # # CycleGAN Model Setup
# # def build_generator():
# #     inputs = layers.Input(shape=(256, 256, 3))
# #     x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
# #     x = layers.ReLU()(x)
# #     x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
# #     x = layers.ReLU()(x)
# #     x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
# #     x = layers.ReLU()(x)
# #     x = layers.Conv2D(512, (4, 4), strides=2, padding='same')(x)
# #     x = layers.ReLU()(x)
# #     # Up-sampling to get back to 256x256
# #     x = layers.Conv2DTranspose(512, (4, 4), strides=2, padding='same')(x)
# #     x = layers.ReLU()(x)
# #     x = layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same')(x)
# #     x = layers.ReLU()(x)
# #     x = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
# #     x = layers.ReLU()(x)
# #     x = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
# #     x = layers.ReLU()(x)
# #     x = layers.Conv2D(3, (4, 4), strides=1, padding='same')(x)  # Keep output at 256x256
# #     x = layers.Activation('tanh')(x)

# #     generator = models.Model(inputs, x)
# #     return generator

# # def build_discriminator():
# #     inputs = layers.Input(shape=(256, 256, 3))
# #     x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = layers.Conv2D(512, (4, 4), strides=2, padding='same')(x)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = layers.Flatten()(x)
# #     x = layers.Dense(1)(x)
# #     x = layers.Activation('sigmoid')(x)

# #     discriminator = models.Model(inputs, x)
# #     return discriminator

# # def build_cyclegan(generator_g, generator_f, discriminator_g, discriminator_f):
# #     real_image = layers.Input(shape=(256, 256, 3))
# #     generated_image = generator_g(real_image)
# #     fake_classification = discriminator_g(generated_image)
# #     cycle_image = generator_f(generated_image)
    
# #     model = models.Model(inputs=real_image, outputs=[fake_classification, cycle_image])
# #     return model

# # # Loss Functions
# # def generator_loss(disc_fake):
# #     return tf.reduce_mean(BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_fake), disc_fake))

# # def discriminator_loss(disc_real, disc_fake):
# #     real_loss = tf.reduce_mean(BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real), disc_real))
# #     fake_loss = tf.reduce_mean(BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_fake), disc_fake))
# #     return (real_loss + fake_loss) * 0.5

# # def cycle_loss(real_image, cycle_image):
# #     real_image = tf.cast(real_image, tf.float32)  # Cast to float32
# #     cycle_image = tf.cast(cycle_image, tf.float32)  # Cast to float32
# #     return tf.reduce_mean(tf.abs(real_image - cycle_image)) * lambda_cycle

# # # Optimizers
# # # Adjust learning rates for better balance
# # generator_optimizer = optimizers.Adam(2e-4, beta_1=0.5)
# # discriminator_optimizer = optimizers.Adam(1e-5, beta_1=0.5)

# # lambda_cycle = 10.0  # Or any other value you want to experiment with
# # lambda_cycle = min(30.0, lambda_cycle + 0.3)  # Increase over epochs

# # # Initialize model components
# # generator_g = build_generator()
# # generator_f = build_generator()
# # discriminator_g = build_discriminator()
# # discriminator_f = build_discriminator()

# # cyclegan_model = build_cyclegan(generator_g, generator_f, discriminator_g, discriminator_f)

# # # Training Loop
# # # After defining the `cyclegan_model`
# # checkpoint_callback.set_model(cyclegan_model)  # Assign the model to the callback

# # # Training Loop
# # epochs = 50
# # batch_size = 4



# # # Prepare dataset from loaded data
# # # Update the function to accept batch_size argument
# # def get_batch_data(emotion, speaker, batch_size=1):
# #     faces = data[emotion][speaker]['faces']
# #     if len(faces) == 0:
# #         print(f"⚠️ No data found for {emotion} - {speaker}")
# #         return None

# #     batch = faces[np.random.randint(0, len(faces), size=batch_size)]
# #     batch = tf.convert_to_tensor(batch, dtype=tf.float32)  # Convert to tensor
# #     return batch



# # # Training step
# # @tf.function
# # def train_step(real_image, emotion):
# #     with tf.GradientTape(persistent=True) as tape:
# #         fake_image = generator_g(real_image)  # Generate fake image
# #         disc_real = discriminator_g(real_image)  # Discriminator output for real image
# #         disc_fake = discriminator_g(fake_image)  # Discriminator output for fake image
# #         cycle_image = generator_f(fake_image)  # Cycle consistency (reconstruct)

# #         # Losses
# #         g_loss = generator_loss(disc_fake)  # Generator loss
# #         cycle = cycle_loss(real_image, cycle_image)  # Cycle loss
# #         total_g_loss = g_loss + (lambda_cycle * cycle)
# #   # Total generator loss

# #         d_loss = discriminator_loss(disc_real, disc_fake)  # Discriminator loss
    
# #     # Compute gradients and apply them
# #     gradients_of_generator = tape.gradient(total_g_loss, generator_g.trainable_variables)
# #     gradients_of_discriminator = tape.gradient(d_loss, discriminator_g.trainable_variables)
    
# #     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_g.trainable_variables))
# #     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_g.trainable_variables))

# #     # Debugging statement
# #     tf.print("Disc Real (Mean):", tf.reduce_mean(disc_real), ", Disc Fake (Mean):", tf.reduce_mean(disc_fake))
# #     tf.print("Generator Loss:", g_loss, ", Total Generator Loss:", total_g_loss)
    
# #     return g_loss, d_loss, total_g_loss

# # import numpy as np
# # import cv2

# # # Track losses
# # import numpy as np
# # import cv2
# # import matplotlib.pyplot as plt

# # generator_losses = []
# # discriminator_losses = []
# # cycle_losses = []

# # def show_generated_images(images):
# #     """Helper function to display generated images."""
# #     fig, axs = plt.subplots(1, len(images), figsize=(12, 4))
# #     if len(images) == 1:
# #         axs = [axs]  # Ensure it's iterable
    
# #     for i, img in enumerate(images):
# #         axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #         axs[i].axis("off")

# #     plt.show()

# # def generate_synthetic_samples(emotion, speaker):
# #     print(f"🛠 Generating samples for Emotion: {emotion}, Speaker: {speaker}")

# #     # Fetch real images from dataset
# #     real_images = get_batch_data(emotion, speaker)  # Removed batch_size

# #     if real_images is None or len(real_images) == 0:
# #         print(f"⚠️ No real images found for [{emotion}] [{speaker}]. Skipping generation.")
# #         return

# #     # 🔍 Debug: Print input shape
# #     print(f"🔍 Real Images Shape: {real_images.shape}")

# #     # Generate synthetic images using the trained generator
# #     synthetic_images = generator_g(real_images)

# #     # 🔍 Debug: Check generator output shape
# #     print(f"🔍 Generator Output Shape: {synthetic_images.shape}")

# #     # Ensure output is valid
# #     if synthetic_images is None or len(synthetic_images) == 0:
# #         print(f"❌ Generator failed to produce images for [{emotion}] [{speaker}].")
# #         return

# #     # Rescale from [-1,1] to [0,255]
# #     synthetic_images = (synthetic_images + 1.0) * 127.5
# #     synthetic_images = np.clip(synthetic_images, 0, 255).astype(np.uint8)

# #     # 🔍 Debug: Display images
# #     show_generated_images(synthetic_images)

# #     # Save images
# #     for i, img in enumerate(synthetic_images):
# #         img_filename = f"synthetic_{emotion}_{speaker}_{i}.jpg"
# #         print(f"📁 Saving: {img_filename}")
# #         cv2.imwrite(img_filename, img)

# #     print(f"✅ Generated {len(synthetic_images)} samples for [{emotion}] [{speaker}]")
# #     return synthetic_images


# # # Start training
# # if check_directory_structure():
# #     data = load_data()

# #     for epoch in range(epochs):
# #         print(f"📢 Epoch {epoch + 1}/{epochs}...")

# #         for emotion in data.keys():
# #             for speaker in data[emotion].keys():
# #                 real_images = get_batch_data(emotion, speaker)

# #                 if real_images is None or len(real_images) == 0:
# #                     print(f"⚠️ Skipping [{emotion}] [{speaker}] due to missing data.")
# #                     continue

# #                 real_images = (real_images / 127.5) - 1.0  # Normalize to [-1, 1]
                
# #                 # Train GAN
# #                 g_loss, d_loss, total_g_loss = train_step(real_images, emotion)

# #                 # Store losses
# #                 generator_losses.append(g_loss)
# #                 discriminator_losses.append(d_loss)
# #                 cycle_losses.append(total_g_loss)

# #                 # Debugging statement
# #                 print(f"[{emotion}] [{speaker}] Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}, Total Generator Loss: {total_g_loss:.4f}")

# #                 # Generate synthetic samples
# #         # generate_synthetic_samples(emotion, speaker)



# #         # Early stopping based on generator loss stagnation
# #         if total_g_loss < min_generator_loss:
# #             min_generator_loss = total_g_loss
# #             best_generator_loss_epoch = epoch
# #             epochs_without_improvement = 0  # Reset counter
# #         else:
# #             epochs_without_improvement += 1


# #         # If no improvement after set patience, stop early
# #         if epochs_without_improvement > max(10, epoch * 0.2):  # More patience in later
# #             print("Early stopping triggered. No improvement in generator loss.")
# #             break

# #         # Save model checkpoint at intervals
# #         if (epoch + 1) % 10 == 0:
# #             checkpoint_callback.on_epoch_end(epoch)

# #     # Save models after training
# #     generator_g.save('generator_g.h5')
# #     generator_f.save('generator_f.h5')
# #     discriminator_g.save('discriminator_g.h5')
# #     discriminator_f.save('discriminator_f.h5')
    
# # print("Training completed.")

# # # After training: Plot the performance metrics
# # plt.figure(figsize=(10, 5))
# # plt.plot(generator_losses, label='Generator Loss')
# # plt.plot(discriminator_losses, label='Discriminator Loss')
# # plt.plot(cycle_losses, label='Cycle Loss')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.title('Performance Metrics per Epoch')
# # plt.show()















































import os
import subprocess
import sys
import cv2
import torch.version
from tqdm import tqdm
import torch
import shutil
import numpy as np
import albumentations as A
import warnings
warnings.simplefilter("ignore")

# 🖥️ Set Paths
STYLEGAN2_ADA_DIR = "D:/MELD/stylegan2-ada-pytorch"
INPUT_DIR = "D:/MELD/output_faces/surprise/Rachel"  # Original images
OUTPUT_DIR = "D:/MELD/stylegan2_dataset/rachel_faces"  # Resized + Augmented images
LMDB_DIR = "D:/MELD/stylegan2_dataset/rachel_lmdb"  # Dataset in LMDB format
TRAIN_DIR = "D:/MELD/stylegan2_training"  # Where the model gets saved

# Print CUDA info
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
print("CUDA Available:", torch.cuda.is_available())
print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Ensure required directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LMDB_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

# ✨ Augmentations for better training
AUGMENTATIONS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2)
])

# 🖼️ Step 1: Resize + Augment Images
def process_images():
    print("\n🔄 Resizing and augmenting images...")
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        print("❌ No images found in INPUT_DIR!")
        return

    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(INPUT_DIR, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Error loading {img_file}")
            continue

        # Resize to 512x512
        img = cv2.resize(img, (512, 512))

        # Save original resized image
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), img)

        # Apply augmentations (5 variations)
        for i in range(5):
            augmented = AUGMENTATIONS(image=img)["image"]
            aug_file = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, aug_file), augmented)

    print("✅ All images processed!")

# 🔄 Step 2: Convert Dataset to ZIP format for StyleGAN2-ADA
def convert_dataset():
    print("\n🔄 Converting dataset to ZIP format...")
    zip_path = os.path.join(LMDB_DIR, "rachel_faces.zip")
    
    if os.path.exists(zip_path):
        os.remove(zip_path)  # Remove old dataset if exists
    
    shutil.make_archive(zip_path.replace(".zip", ""), 'zip', OUTPUT_DIR)
    print("✅ Dataset converted to ZIP format!")

# 🚀 Step 3: Train StyleGAN2-ADA
# 🚀 Step 3: Train StyleGAN2-ADA (Live Updates)
def train_stylegan2():
    print("\n🚀 Starting StyleGAN2-ADA training...\n")

    cmd = [
    sys.executable,
    os.path.join(STYLEGAN2_ADA_DIR, "train.py"),
    "--outdir", TRAIN_DIR,
    "--data", os.path.join(LMDB_DIR, "rachel_faces.zip"),
    "--gpus", "1",
    "--batch", "8",  # Adjust based on GPU VRAM
    "--kimg", "500",   # Train for 500 kimg (adjust as needed)
    "--mirror", "1",    # Enable horizontal mirroring
    "--snap", "10",     # Save snapshots every 10 ticks
]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)

        # ✅ Read output line-by-line in real-time
        for line in iter(process.stdout.readline, ""):
            if "tick" in line:
                print(f"\033[94m📌 {line.strip()}\033[0m")  # Blue color for progress
            elif "kimg" in line:
                print(f"\033[92m🖼️ {line.strip()}\033[0m")  # Green color for image training updates
            elif "CUDA" in line:
                print(f"\033[93m⚡ {line.strip()}\033[0m")  # Yellow for CUDA info
            else:
                print(line.strip())  # Default output
            sys.stdout.flush()  # Force output to appear instantly

        process.wait()
        print("\n✅ Training completed! Check logs for details.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Training error: {e}")


# 🏁 Run everything
if __name__ == "__main__":
    print("\n🚀 Starting Full StyleGAN2-ADA Pipeline 🚀")
    process_images()
    convert_dataset()
    train_stylegan2()
