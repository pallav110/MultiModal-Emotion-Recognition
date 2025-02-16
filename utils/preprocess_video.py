# import cv2
# import mediapipe as mp

# print("🚀 Testing Mediapipe and OpenCV Initialization...")

# # Mediapipe Models
# mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
# mp_pose = mp.solutions.pose.Pose()

# # OpenCV Face Detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# if face_cascade.empty():
#     print("❌ OpenCV Haar Cascade failed to load.")
# else:
#     print("✅ OpenCV Haar Cascade loaded successfully.")

# print("✅ Mediapipe models initialized successfully.")




# import os
# import cv2

# VIDEO_BASE_DIR = "D:/MELD/MELD.Raw/"
# DATASET_PATHS = {
#     "train": "train/train_splits",
#     "dev": "dev/dev_splits_complete"
# }

# # Check videos
# for dataset, subfolder in DATASET_PATHS.items():
#     video_folder = os.path.join(VIDEO_BASE_DIR, subfolder)
#     print(f"\n🔍 Checking videos in: {video_folder}")

#     video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

#     if len(video_files) == 0:
#         print(f"⚠️ No videos found in {video_folder}")
#         continue

#     print(f"✅ Found {len(video_files)} videos. First 5: {video_files[:5]}")

#     # Test reading the first video
#     test_video_path = os.path.join(video_folder, video_files[0])
#     cap = cv2.VideoCapture(test_video_path)

#     if not cap.isOpened():
#         print(f"❌ ERROR: Cannot open {test_video_path}")
#     else:
#         print(f"✅ Successfully opened: {test_video_path}")
    
#     cap.release()





# import cv2
# import numpy as np
# import mediapipe as mp

# #import cv2
# import numpy as np
# import mediapipe as mp
# import cv2
# import numpy as np
# import mediapipe as mp
# import cv2
# import numpy as np
# import mediapipe as mp

# # Load Mediapipe models
# mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
# mp_pose = mp.solutions.pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# # Load OpenCV face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# def extract_features(video_path, max_frames=10):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"❌ ERROR: Cannot open {video_path}")
#         return None  

#     frame_count = 0
#     while frame_count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"⚠️ Cannot read frame {frame_count}, skipping.")
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # 🔹 Enhance contrast & brightness
#         frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.5, beta=50)

#         # 🔹 Detect faces
#         faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

#         if len(faces) > 0:
#             print(f"✅ OpenCV detected {len(faces)} face(s) in Frame {frame_count}")

#             x, y, w, h = faces[0]
#             padding = int(0.2 * w)  # Expand bounding box
#             x = max(0, x - padding)
#             y = max(0, y - padding)
#             w = min(frame.shape[1] - x, w + 2 * padding)
#             h = min(frame.shape[0] - y, h + 2 * padding)

#             face_crop = frame_rgb[y:y+h, x:x+w]
#             face_crop_resized = cv2.resize(face_crop, (224, 224))

#             # 🔹 Face landmarks detection
#             face_results = mp_face.process(face_crop_resized)

#             if face_results.multi_face_landmarks:
#                 print(f"🎯 Frame {frame_count}: Face landmarks detected!")
#                 # Draw face landmarks
#                 for landmark in face_results.multi_face_landmarks[0].landmark:
#                     lx, ly = int(landmark.x * w + x), int(landmark.y * h + y)
#                     cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)  # Green dots
#             else:
#                 print(f"⚠️ Frame {frame_count}: No face landmarks detected.")

#         else:
#             print(f"⚠️ Frame {frame_count}: No face detected!")

#         # 🔹 Pose landmarks detection
#         pose_results = mp_pose.process(frame_rgb)

#         if pose_results.pose_landmarks:
#             print(f"🎯 Frame {frame_count}: Pose landmarks detected!")
#             mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
#         else:
#             print(f"⚠️ Frame {frame_count}: No pose landmarks detected.")

#         # 🔹 Show processed frame (SLOWER)
#         cv2.imshow("Processed Frame", frame)
#         key = cv2.waitKey(200)  # 🕒 Delay of 200ms per frame

#         if key == ord('q'):  # Press 'q' to quit
#             break  

#         frame_count += 1

#     cap.release()
#     cv2.destroyAllWindows()
#     print("✅ Feature extraction test complete.")

# # Test on one video
# VIDEO_PATH = "D:/MELD/MELD.Raw/train/train_splits/dia0_utt0.mp4"
# extract_features(VIDEO_PATH)














# import os
# import cv2
# import time
# import mediapipe as mp

# # 📁 Paths
# VIDEO_BASE_DIR = "D:/MELD/MELD.Raw/"
# DATASET_PATH = "train/train_splits"
# video_folder = os.path.join(VIDEO_BASE_DIR, DATASET_PATH)

# # 🔍 Get Video Files (Only First 5)
# video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")][:5]

# print(f"\n📂 Found {len(video_files)} videos: {video_files}")

# # 🎭 Initialize MediaPipe Modules
# mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
# mp_pose = mp.solutions.pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# # 🟢 OpenCV Face Detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# def extract_features(frame, frame_count):
#     """Extract face & pose features from a single frame."""
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # 🔹 Enhance contrast & brightness
#     frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.5, beta=50)

#     # 🔹 Detect faces using OpenCV
#     faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

#     if len(faces) > 0:
#         print(f"✅ OpenCV detected {len(faces)} face(s) in Frame {frame_count}")

#         x, y, w, h = faces[0]
#         padding = int(0.2 * w)  # Expand bounding box
#         x = max(0, x - padding)
#         y = max(0, y - padding)
#         w = min(frame.shape[1] - x, w + 2 * padding)
#         h = min(frame.shape[0] - y, h + 2 * padding)

#         face_crop = frame_rgb[y:y+h, x:x+w]
#         face_crop_resized = cv2.resize(face_crop, (224, 224))

#         # 🔹 Face landmarks detection
#         face_results = mp_face_mesh.process(face_crop_resized)

#         if face_results.multi_face_landmarks:
#             print(f"🎯 Frame {frame_count}: Face landmarks detected!")
#             # Draw face landmarks
#             for landmark in face_results.multi_face_landmarks[0].landmark:
#                 lx, ly = int(landmark.x * w + x), int(landmark.y * h + y)
#                 cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)  # Green dots
#         else:
#             print(f"⚠️ Frame {frame_count}: No face landmarks detected.")
#     else:
#         print(f"⚠️ Frame {frame_count}: No face detected!")

#     # 🔹 Pose landmarks detection
#     pose_results = mp_pose.process(frame_rgb)

#     if pose_results.pose_landmarks:
#         print(f"🎯 Frame {frame_count}: Pose landmarks detected!")
#         mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
#     else:
#         print(f"⚠️ Frame {frame_count}: No pose landmarks detected.")

#     return frame  # Return frame with features drawn

# def process_video(video_path, delay=1.0, max_frames=10):
#     """Plays a video and extracts features while ensuring frames sync properly."""
#     print(f"\n📽️ Opening: {video_path}")
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"❌ ERROR: Cannot open {video_path}")
#         return

#     frame_count = 0

#     while frame_count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"⚠️ End of Video: {video_path}")
#             break

#         print(f"🖼️ Displaying Frame {frame_count}")
#         frame_count += 1

#         # ✅ Extract Features (Face + Pose)
#         processed_frame = extract_features(frame.copy(), frame_count)

#         # 🔹 Show Processed Frame
#         cv2.imshow("Processed Frame", processed_frame)
#         time.sleep(delay)  # 🐢 Slow down playback

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("❌ Stopping video playback...")
#             break  # Quit on 'q' key

#     cap.release()
#     cv2.destroyAllWindows()

# # 🚀 Process & Extract Features from Each Video
# for idx, video in enumerate(video_files):
#     video_path = os.path.join(video_folder, video)
#     print(f"\n📂 Processing: {video_path}")
    
#     process_video(video_path, delay=0.5, max_frames=10)  # Play & Extract Features

# # 🔄 Clean Up
# mp_face_mesh.close()
# mp_pose.close()

# print("\n✅ Combined Video Processing + Feature Extraction Test Complete!")























import os
import cv2
import numpy as np
import mediapipe as mp
import json
from tqdm import tqdm

# 📁 Paths
VIDEO_BASE_DIR = "D:/MELD/MELD.Raw/"
OUTPUT_DIR = "D:/MELD/processed/video_features/"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.json")
COMBINED_NPY_FILE = os.path.join(OUTPUT_DIR, "all_features.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📂 Dataset Paths
DATASET_PATHS = {
    "train": "train/train_splits",
    "dev": "dev/dev_splits_complete"
}

# 🔹 Initialize Mediapipe Models
print("🚀 Initializing Mediapipe models...")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)
mp_pose = mp.solutions.pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 📄 Logging Setup
video_log = {}

def extract_features(frame):
    """Extracts face and pose keypoints from a given frame, ensuring consistent output shape."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔹 Enhance contrast & brightness
    frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.3, beta=40)

    # 🔹 Face Detection
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Default to zero-filled arrays
    face_keypoints = np.zeros((468, 3), dtype=np.float32)
    body_keypoints = np.zeros((33, 3), dtype=np.float32)

    face_crop_resized = None  # Ensure it's always defined

    if len(faces) > 0:
        x, y, w, h = faces[0]
        padding = int(0.2 * w)  # Expand bounding box
        x, y, w, h = max(0, x - padding), max(0, y - padding), min(frame.shape[1] - x, w + 2 * padding), min(frame.shape[0] - y, h + 2 * padding)

        face_crop = frame_rgb[y:y+h, x:x+w]

        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:  # Ensure valid crop
            face_crop_resized = cv2.resize(face_crop, (224, 224))

    # 🔹 Face Mesh Processing
    if face_crop_resized is not None:
        face_results = mp_face_mesh.process(face_crop_resized)
        if face_results.multi_face_landmarks:
            detected_landmarks = face_results.multi_face_landmarks[0].landmark
            num_detected = min(len(detected_landmarks), 468)  # Ensure we don't exceed 468

            for i in range(num_detected):
                face_keypoints[i] = [detected_landmarks[i].x, detected_landmarks[i].y, detected_landmarks[i].z]

    # 🔹 Pose Processing
    pose_results = mp_pose.process(frame_rgb)
    if pose_results.pose_landmarks:
        for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
            body_keypoints[i] = [landmark.x, landmark.y, landmark.z]

    # 🔹 Flatten & Ensure Consistent Shape
    feature_vector = np.concatenate([face_keypoints.flatten(), body_keypoints.flatten()])
    return feature_vector


def process_video(video_path, max_frames=10):
    """Processes a video, extracts features, and logs the results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open {video_path}")
        return None

    frame_count = 0
    frame_features = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends early

        frame_count += 1
        features = extract_features(frame)
        frame_features.append(features)

    cap.release()

    if len(frame_features) == 0:
        return None

    return np.array(frame_features, dtype=np.float32)

# 🚀 **Process All Videos**
all_features = []
for dataset, subfolder in DATASET_PATHS.items():
    video_folder = os.path.join(VIDEO_BASE_DIR, subfolder)
    output_folder = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    print(f"\n🔍 Found {len(video_files)} videos in {dataset}")

    for idx, video in enumerate(video_files):
        print(f"📌 Processing {dataset}: {idx+1}/{len(video_files)} - {video}")

        video_path = os.path.join(video_folder, video)
        output_path = os.path.join(output_folder, os.path.splitext(video)[0] + ".npy")

        try:
            features = process_video(video_path, max_frames=10)
            if features is not None:
                np.save(output_path, features)
                all_features.append(features)

                # 📝 Log video details
                video_log[video] = {
                    "frames_processed": features.shape[0],
                    "feature_shape": features.shape
                }
            else:
                video_log[video] = {"error": "No features extracted"}

        except Exception as e:
            print(f"❌ ERROR processing {video}: {str(e)}")
            video_log[video] = {"error": str(e)}


# 📝 Save log file
with open(LOG_FILE, "w") as log_file:
    json.dump(video_log, log_file, indent=4)

# 📌 Combine all features into one file
if all_features:
    combined_features = np.concatenate(all_features, axis=0)
    np.save(COMBINED_NPY_FILE, combined_features)
    print(f"✅ Combined features saved: {COMBINED_NPY_FILE}")

print("🎉 Video processing complete!")
