import cv2
import numpy as np
import os
import random

# Paths
RAW_VIDEO_DIR = "D:/MELD/MELD.Raw/video"
AUGMENTED_VIDEO_DIR = "D:/MELD/processed/augmented_video"

# Ensure output directory exists
os.makedirs(AUGMENTED_VIDEO_DIR, exist_ok=True)

# List of video files (example: .mp4 or .avi)
video_files = [os.path.join(RAW_VIDEO_DIR, f) for f in os.listdir(RAW_VIDEO_DIR) if f.endswith('.mp4')]

# Function to flip the video frames horizontally
def flip_frame(frame):
    """Flip the video frame horizontally."""
    return cv2.flip(frame, 1)

# Function to rotate the video frame by a random angle
def rotate_frame(frame):
    """Rotate the video frame by a random angle."""
    angle = random.randint(-30, 30)  # Random angle between -30 and 30 degrees
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)  # Rotation matrix
    rotated_frame = cv2.warpAffine(frame, matrix, (width, height))
    return rotated_frame

# Function to interpolate between two frames (for smoother transitions)
def interpolate_frames(frame1, frame2):
    """Interpolate between two frames to create a smoother transition."""
    alpha = random.uniform(0.3, 0.7)  # Random interpolation factor
    interpolated_frame = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)
    return interpolated_frame

# Function to augment the video
def augment_video(video_file):
    """Augment the video by applying flipping, rotation, and frame interpolation."""
    # Read video file using OpenCV
    cap = cv2.VideoCapture(video_file)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of frames
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

    augmented_frames = []
    
    # Read frames and augment
    prev_frame = None
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply frame augmentations
        if random.choice([True, False]):
            frame = flip_frame(frame)
        if random.choice([True, False]):
            frame = rotate_frame(frame)
        if prev_frame is not None and random.choice([True, False]):
            frame = interpolate_frames(prev_frame, frame)

        # Store the augmented frame
        augmented_frames.append(frame)
        
        # Update previous frame
        prev_frame = frame

    cap.release()
    return augmented_frames, fps, width, height

# Function to save the augmented video
def save_augmented_video(augmented_frames, fps, width, height, original_video_file):
    """Save the augmented frames as a new video file."""
    base_filename = os.path.basename(original_video_file)
    augmented_filename = f"augmented_{base_filename}"

    # Create a VideoWriter object to save the video
    output_path = os.path.join(AUGMENTED_VIDEO_DIR, augmented_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each augmented frame to the video
    for frame in augmented_frames:
        out.write(frame)

    out.release()
    print(f"Saved augmented video: {output_path}")

# Main process: Loop through video files and augment them
for video_file in video_files:
    print(f"Processing {video_file}...")
    
    # Perform video augmentation
    augmented_frames, fps, width, height = augment_video(video_file)
    
    # Save the augmented video
    save_augmented_video(augmented_frames, fps, width, height, video_file)

print("Video augmentation complete!")
