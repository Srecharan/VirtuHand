import subprocess
import os

def combine_videos_side_by_side():
    # Define paths
    left_video = "/home/rex/real_time_hand_gesture_v1/unity.mp4"
    right_video = "/home/rex/real_time_hand_gesture_v1/hand.mp4"
    output_path = "/home/rex/real_time_hand_gesture_v1/final_combined.mp4"
    
    # Check if input files exist
    if not os.path.exists(left_video):
        print(f"Error: Unity video not found at {left_video}")
        return
    if not os.path.exists(right_video):
        print(f"Error: Hand tracking video not found at {right_video}")
        return
    
    # FFmpeg command
    command = [
        'ffmpeg',
        '-i', left_video,
        '-i', right_video,
        '-filter_complex',
        '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0',
        '-c:v', 'libx264',
        '-crf', '18',  # High quality (lower = better quality)
        '-preset', 'slow',  # Better compression
        output_path
    ]
    
    print("Starting video combination...")
    try:
        # Run FFmpeg command
        subprocess.run(command, check=True)
        print(f"\nSuccess! Combined video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError combining videos: {e}")

if __name__ == "__main__":
    combine_videos_side_by_side()