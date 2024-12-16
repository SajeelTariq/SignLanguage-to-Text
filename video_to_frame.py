import cv2
import os

# Set the path to the input video file
video_path = 'D:/5th semester/AI&ES/TestProject333333333/Videos/VideoCombined/Class.mp4'

# Directory to save the extracted frames
output_dir = 'D:/5th semester/AI&ES/TestProject333333333/data/19'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while True:
    # Read the next frame from the video
    success, frame = video_capture.read()
    
    # Break the loop if there are no more frames
    if not success:
        break
    
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    
    # Save the frame as a JPEG image
    cv2.imwrite(frame_filename, frame)
    
    print(f"Saved {frame_filename}")
    
    frame_count += 1

video_capture.release()
print("All frames have been saved successfully.")
