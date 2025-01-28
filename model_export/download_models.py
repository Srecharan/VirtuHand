import os
import requests
from pathlib import Path
import shutil

def create_output_directory():
    output_dir = os.path.join(os.path.dirname(__file__), 'exported_models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def download_file(url, output_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"Downloaded to {output_path}")

def main():
    output_dir = create_output_directory()
    print(f"Output directory created at: {output_dir}")
    
    # Updated MediaPipe model URLs
    model_urls = {
        'palm_detection.onnx': 'https://github.com/google/mediapipe/raw/master/mediapipe/modules/hand_landmark/hand_landmark_full.onnx',
        'hand_landmark.onnx': 'https://github.com/google/mediapipe/raw/master/mediapipe/modules/hand_landmark/hand_landmark_lite.onnx'
    }
    
    for filename, url in model_urls.items():
        output_path = os.path.join(output_dir, filename)
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            continue
    
    print("\nModel download complete!")
    print("Please copy the downloaded models from the 'exported_models' directory")
    print("to your Unity project's 'Assets/Resources/Models' folder.")

if __name__ == "__main__":
    main()