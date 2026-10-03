import os
import cv2
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.config import load_config

class DatasetInspector:
    """Utility to inspect traffic datasets, video files, and image annotations."""

    def __init__(self, raw_dir: Optional[str] = None):
        cfg = load_config()
        self.raw_dir = raw_dir or cfg.paths.raw_data_dir
        os.makedirs(self.raw_dir, exist_ok=True)

    def inspect_raw_directory(self) -> Dict[str, Any]:
        """Inspects all files in data/raw directory."""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.raw_dir, ext)))
            video_files.extend(glob.glob(os.path.join(self.raw_dir, "**", ext), recursive=True))
            
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.raw_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.raw_dir, "**", ext), recursive=True))

        json_files = glob.glob(os.path.join(self.raw_dir, "**", "*.json"), recursive=True)

        summary = {
            "num_video_files": len(video_files),
            "video_files": [os.path.basename(f) for f in video_files],
            "num_image_files": len(image_files),
            "num_json_annotations": len(json_files),
            "video_details": []
        }

        for vf in video_files:
            cap = cv2.VideoCapture(vf)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = frame_count / fps if fps > 0 else 0
                summary["video_details"].append({
                    "filename": os.path.basename(vf),
                    "path": vf,
                    "dimensions": f"{width}x{height}",
                    "fps": round(fps, 2),
                    "total_frames": frame_count,
                    "duration_seconds": round(duration_sec, 2)
                })
            cap.release()

        return summary

    def generate_synthetic_traffic_video(self, output_path: str = "data/raw/sample_traffic.mp4", duration_sec: int = 15, fps: int = 30) -> str:
        """Generates a synthetic traffic video simulating vehicles moving on a highway with varying density."""
        width, height = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        num_frames = duration_sec * fps
        
        # Vehicle structures: [x, y, speed, type, color, size]
        np.random.seed(42)
        vehicles = []
        
        # Define 3 lanes
        lane_y_positions = [250, 400, 550]
        vehicle_types = ['car', 'truck', 'bus', 'motorcycle']
        type_colors = {
            'car': (0, 165, 255),        # Orange
            'truck': (255, 0, 0),        # Blue
            'bus': (0, 255, 0),          # Green
            'motorcycle': (255, 255, 0)  # Cyan
        }
        type_sizes = {
            'car': (80, 40),
            'truck': (140, 50),
            'bus': (130, 55),
            'motorcycle': (40, 20)
        }

        # Initialize synthetic vehicles
        for i in range(15):
            vtype = np.random.choice(vehicle_types, p=[0.6, 0.15, 0.15, 0.1])
            lane_idx = np.random.randint(0, 3)
            y = lane_y_positions[lane_idx] + np.random.randint(-15, 15)
            x = np.random.randint(0, width)
            speed = np.random.uniform(4.0, 12.0)
            vehicles.append({
                'id': i + 1,
                'type': vtype,
                'x': float(x),
                'y': float(y),
                'speed': speed,
                'color': type_colors[vtype],
                'size': type_sizes[vtype]
            })

        for frame_idx in range(num_frames):
            # Dark asphalt background
            frame = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)

            # Draw highway lanes
            cv2.rectangle(frame, (0, 180), (width, 650), (60, 60, 60), -1)
            # Solid white top/bottom road boundaries
            cv2.line(frame, (0, 180), (width, 180), (255, 255, 255), 4)
            cv2.line(frame, (0, 650), (width, 650), (255, 255, 255), 4)

            # Dashed yellow lane divider
            dash_length = 30
            for x_dash in range((frame_idx * 5) % (dash_length * 2) - dash_length * 2, width, dash_length * 2):
                cv2.line(frame, (max(0, x_dash), 325), (min(width, x_dash + dash_length), 325), (0, 255, 255), 2)
                cv2.line(frame, (max(0, x_dash), 485), (min(width, x_dash + dash_length), 485), (0, 255, 255), 2)

            # Draw HUD info
            cv2.putText(frame, f"SYNTHETIC TRAFFIC STREAM | Frame: {frame_idx}/{num_frames} | FPS: {fps}", 
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Update & draw vehicles
            for v in vehicles:
                v['x'] += v['speed']
                # Loop around when reaching screen edge
                if v['x'] > width + 100:
                    v['x'] = -150
                    v['speed'] = np.random.uniform(4.0, 12.0)

                vx = int(v['x'])
                vy = int(v['y'])
                vw, vh = v['size']

                # Vehicle body rectangle
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), v['color'], -1)
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (255, 255, 255), 2)
                
                # Wheels / details
                cv2.circle(frame, (vx + 15, vy + vh), 5, (10, 10, 10), -1)
                cv2.circle(frame, (vx + vw - 15, vy + vh), 5, (10, 10, 10), -1)

            out.write(frame)

        out.release()
        return output_path

if __name__ == "__main__":
    inspector = DatasetInspector()
    info = inspector.inspect_raw_directory()
    print("=== RAW DATASET INSPECTION ===")
    print(f"Video Files Found: {info['num_video_files']}")
    if info['num_video_files'] == 0:
        print("No video found in data/raw. Generating synthetic sample video...")
        gen_path = inspector.generate_synthetic_traffic_video()
        print(f"Generated synthetic traffic video at: {gen_path}")
        info = inspector.inspect_raw_directory()
    
    print("\nUpdated Dataset Summary:")
    print(info)
