# src/gesture/train_dynamic_gestures.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.gesture.dynamic_gesture import GRUGestureModel, DynamicGestureType
from src.gesture.detector import HandDetector
import pyrealsense2 as rs

class HandGestureDataCollector:
    def __init__(self, save_dir="dataset/dynamic_gestures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Initialize hand detector
        self.hand_detector = HandDetector(max_hands=1)
        
        # Initialize gesture sequences
        self.current_sequence = []
        self.sequence_length = 30  # 1 second at 30 FPS
        
    def collect_data(self):
        """Collect training data for dynamic gestures"""
        try:
            self.pipeline.start(self.config)
            
            for gesture_type in DynamicGestureType:
                if gesture_type == DynamicGestureType.NONE:
                    continue
                    
                gesture_dir = self.save_dir / gesture_type.name.lower()
                gesture_dir.mkdir(exist_ok=True)
                
                print("\n" + "="*50)
                print(f"Collecting data for: {gesture_type.name}")
                print("="*50)
                print("\nInstructions:")
                print(f"- Perform {gesture_type.name} gesture")
                print("- Press SPACE to start each recording")
                print("- Press ESC to skip to next gesture type")
                print("- Press Ctrl+C to quit completely")
                print("\nNeeded: 20 recordings")
                
                sequence_count = 0
                while sequence_count < 20:  # Collect 20 sequences per gesture
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    
                    if color_frame:
                        preview = np.asanyarray(color_frame.get_data())
                        cv2.putText(preview,
                                f"Ready for {gesture_type.name} - Sequence {sequence_count + 1}/20",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(preview,
                                "Press SPACE to start recording or ESC to skip",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Preview', preview)
                        
                        key = cv2.waitKey(1)
                        if key == 27:  # ESC
                            break
                        elif key == 32:  # SPACE
                            sequence = self.record_sequence(sequence_count, gesture_type)
                            if sequence:
                                self.save_sequence(sequence, gesture_dir / f"sequence_{sequence_count}.json")
                                sequence_count += 1
                    
                print(f"\nCollected {sequence_count} sequences for {gesture_type.name}")
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
                
    def record_sequence(self, sequence_num, gesture_type):
        """Record a single gesture sequence"""
        sequence = []
        
        # Countdown before recording
        preview_image = None
        for countdown in range(3, 0, -1):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                preview_image = np.asanyarray(color_frame.get_data())
                # Add big countdown number
                cv2.putText(preview_image, 
                        str(countdown), 
                        (preview_image.shape[1]//2 - 50, preview_image.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
                cv2.putText(preview_image,
                        "Get ready to perform gesture",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Recording', preview_image)
                cv2.waitKey(1000)  # Wait 1 second between countdown numbers
        
        # Show "RECORDING NOW!" for 0.5 seconds
        if preview_image is not None:
            cv2.putText(preview_image,
                    "RECORDING NOW!",
                    (preview_image.shape[1]//2 - 150, preview_image.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('Recording', preview_image)
            cv2.waitKey(500)

        print(f"\nRecording {gesture_type.name} - Sequence {sequence_num + 1}/20")
        start_time = time.time()
        
        # Recording loop
        progress_bar_length = 30
        while len(sequence) < self.sequence_length:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Detect hand landmarks
            hands = self.hand_detector.detect_hands(color_image, depth_image)
            
            if hands:
                # Extract landmark positions
                landmarks = hands[0].landmarks
                landmark_data = {str(idx): [lm.x, lm.y, lm.z] for idx, lm in landmarks.items()}
                sequence.append(landmark_data)
                
                # Create progress bar
                progress = len(sequence) / self.sequence_length
                filled_length = int(progress_bar_length * progress)
                progress_bar = '=' * filled_length + '-' * (progress_bar_length - filled_length)
                
                # Visualize recording with enhanced feedback
                annotated_image = self.hand_detector.draw_landmarks(color_image, hands)
                cv2.putText(annotated_image, 
                        f"Recording {gesture_type.name} - Sequence {sequence_num + 1}/20", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_image,
                        f"Progress: [{progress_bar}] {int(progress*100)}%",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Recording', annotated_image)
                
                # Handle key presses
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    return None
                    
        # Show "RECORDED!" for 1 second
        if len(sequence) == self.sequence_length:
            cv2.putText(annotated_image,
                    "RECORDED SUCCESSFULLY!",
                    (annotated_image.shape[1]//2 - 200, annotated_image.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Recording', annotated_image)
            cv2.waitKey(1000)
        
        return sequence if len(sequence) == self.sequence_length else None

    @staticmethod
    def save_sequence(sequence, filepath):
        """Save sequence data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(sequence, f)

class DynamicGestureDataset(Dataset):
    def __init__(self, data_dir, sequence_length=30):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        # Load all sequences
        for gesture_dir in self.data_dir.iterdir():
            if gesture_dir.is_dir():
                try:
                    gesture_type = DynamicGestureType[gesture_dir.name.upper()]
                    
                    for sequence_file in gesture_dir.glob("*.json"):
                        with open(sequence_file, 'r') as f:
                            sequence = json.load(f)
                            self.sequences.append(sequence)
                            # Use gesture_type.value instead of just gesture_type
                            self.labels.append(gesture_type.value)
                except KeyError:
                    continue
                        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert sequence to tensor
        sequence_tensor = []
        for frame in sequence:
            landmarks = []
            for i in range(21):  # 21 hand landmarks
                if str(i) in frame:
                    landmarks.extend(frame[str(i)])
                else:
                    landmarks.extend([0, 0, 0])
            sequence_tensor.append(landmarks)
            
        return torch.FloatTensor(sequence_tensor), torch.tensor(label, dtype=torch.long)

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Print dataset info
    print(f"Training on device: {device}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Debug info
            if batch_idx == 0:
                print(f"\nSequences shape: {sequences.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100.0 * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_dynamic_gesture_model.pth")
            
def main():
    while True:
        print("\nOptions:")
        print("1. Use existing recorded data")
        print("2. Record new data")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")

    if choice == '2':
        # Record new data
        collector = HandGestureDataCollector()
        collector.collect_data()
    else:
        print("\nUsing existing recorded data...")
        
    # Check if we have enough data to proceed
    dataset_path = Path("dataset/dynamic_gestures")
    if not dataset_path.exists():
        print("Error: No dataset found at", dataset_path)
        return
        
    # Count existing sequences
    gesture_counts = {}
    for gesture_dir in dataset_path.iterdir():
        if gesture_dir.is_dir():
            count = len(list(gesture_dir.glob("*.json")))
            gesture_counts[gesture_dir.name] = count
            
    print("\nFound existing recordings:")
    for gesture, count in gesture_counts.items():
        print(f"{gesture}: {count} sequences")
        
    proceed = input("\nProceed with training? (y/n): ")
    if proceed.lower() != 'y':
        return
    
    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = DynamicGestureDataset("dataset/dynamic_gestures")
    
    if len(dataset) == 0:
        print("Error: No valid sequences found in the dataset")
        return
        
    print(f"Total sequences in dataset: {len(dataset)}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create and train model
    print("\nStarting model training...")
    model = GRUGestureModel()
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()