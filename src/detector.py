import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import argparse
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ultralytics import YOLO
from src.config import load_config, DetectorConfig

class YOLODetector:
    """YOLO Object Detector wrapper for traffic monitoring."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = load_config().detector
        self.config = config
        
        print(f"[INFO] Loading YOLO model: {self.config.model_name}")
        self.model = YOLO(self.config.model_name)
        self.target_classes = self.config.target_classes

    def detect_frame(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Runs YOLO inference on a single BGR image frame."""
        conf = conf_threshold if conf_threshold is not None else self.config.conf_threshold
        
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False
        )
        
        annotated_frame = frame.copy()
        detections = []
        class_counts = {name: 0 for name in set(self.target_classes.values())}

        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                
                class_name = self.target_classes.get(cls_id, self.model.names.get(cls_id, "unknown"))
                
                det = {
                    "bbox": [round(c, 2) for c in xyxy],
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4)
                }
                detections.append(det)

                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # Draw bounding box
                x1, y1, x2, y2 = [int(c) for c in xyxy]
                color = self._get_class_color(cls_id)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return {
            "annotated_frame": annotated_frame,
            "detections": detections,
            "class_counts": class_counts,
            "total_count": len(detections)
        }

    def process_video_file(self, video_path: str, output_path: str = "data/processed/test_detection_output.mp4", max_frames: int = 300, conf_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Runs YOLO detection frame-by-frame on a real video and saves annotated output."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Real traffic video file missing at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_detections_sum = 0
        overall_class_counts: Dict[str, int] = {}
        frame_count = 0

        start_time = time.time()

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            res = self.detect_frame(frame, conf_threshold=conf_threshold)
            ann_frame = res["annotated_frame"]
            
            # Overlay frame number and count HUD
            hud_text = f"Frame: {frame_count}/{max_frames} | Detections: {res['total_count']}"
            cv2.putText(ann_frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            out.write(ann_frame)

            total_detections_sum += res["total_count"]
            for cls_name, cnt in res["class_counts"].items():
                overall_class_counts[cls_name] = overall_class_counts.get(cls_name, 0) + cnt

        cap.release()
        out.release()

        elapsed_sec = max(0.001, time.time() - start_time)
        processing_fps = round(frame_count / elapsed_sec, 2)
        avg_detections_per_frame = round(total_detections_sum / max(1, frame_count), 2)

        return {
            "video_path": video_path,
            "resolution": f"{width}x{height}",
            "video_fps": round(fps, 2),
            "frames_processed": frame_count,
            "processing_time_sec": round(elapsed_sec, 2),
            "processing_fps": processing_fps,
            "model_used": self.config.model_name,
            "confidence_threshold": conf_threshold or self.config.conf_threshold,
            "total_detections": total_detections_sum,
            "avg_detections_per_frame": avg_detections_per_frame,
            "detected_class_counts": overall_class_counts,
            "output_path": output_path
        }

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        colors = [
            (255, 102, 102), (255, 178, 102), (0, 204, 255), (255, 102, 255),
            (102, 255, 102), (102, 102, 255), (255, 255, 102), (204, 102, 255)
        ]
        return colors[class_id % len(colors)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Real Video Detection Processor")
    parser.add_argument("--video", type=str, default="data/raw/videos/test_traffic.mp4", help="Path to real traffic video file")
    parser.add_argument("--output", type=str, default="data/processed/test_detection_output.mp4", help="Path for annotated video output")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to process")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Real traffic video not found at: {args.video}")
        print("Please place a traffic video at data/raw/videos/test_traffic.mp4")
    else:
        detector = YOLODetector()
        stats = detector.process_video_file(args.video, args.output, max_frames=args.max_frames, conf_threshold=args.conf)
        print("\n=== YOLO VIDEO DETECTION STATISTICS ===")
        print(stats)
