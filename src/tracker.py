import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import argparse
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ultralytics import YOLO
from src.config import load_config, TrackerConfig, DetectorConfig

class VehicleTracker:
    """ByteTrack Multi-Object Tracker wrapper utilizing persistent vehicle tracking."""

    def __init__(self, detector_config: Optional[DetectorConfig] = None, tracker_config: Optional[TrackerConfig] = None):
        cfg = load_config()
        self.det_config = detector_config or cfg.detector
        self.track_config = tracker_config or cfg.tracker
        
        self.model = YOLO(self.det_config.model_name)
        self.target_classes = self.det_config.target_classes

        # Active track trajectory history: track_id -> list of (cx, cy, frame_idx)
        self.track_history: Dict[int, List[Tuple[float, float, int]]] = {}
        self.seen_vehicle_ids = set()

    def reset_tracker(self):
        """Resets active track history and ID counters."""
        self.track_history.clear()
        self.seen_vehicle_ids.clear()

    def track_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict[str, Any]:
        """Runs YOLO + ByteTrack multi-object tracking on a frame."""
        results = self.model.track(
            source=frame,
            conf=self.det_config.conf_threshold,
            iou=self.det_config.iou_threshold,
            device=self.det_config.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        annotated_frame = frame.copy()
        tracked_objects = []
        class_counts = {name: 0 for name in set(self.target_classes.values())}

        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            result = results[0]
            boxes = result.boxes
            
            xyxys = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for xyxy, track_id, cls_id, conf in zip(xyxys, track_ids, cls_ids, confs):
                class_name = self.target_classes.get(int(cls_id), self.model.names.get(int(cls_id), "unknown"))
                
                x1, y1, x2, y2 = [float(c) for c in xyxy]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                # Update trajectory history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append((cx, cy, frame_idx))
                
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)

                # Calculate Relative Speed Proxy in pixels/frame
                relative_speed_proxy_px_per_frame = 0.0
                if len(self.track_history[track_id]) >= 2:
                    p1 = self.track_history[track_id][-2]
                    p2 = self.track_history[track_id][-1]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    relative_speed_proxy_px_per_frame = float(np.sqrt(dx*dx + dy*dy))

                # Track unique vehicle IDs
                if class_name in ["car", "bus", "truck", "motorcycle", "bicycle"]:
                    self.seen_vehicle_ids.add(int(track_id))

                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                tracked_obj = {
                    "track_id": int(track_id),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": round(float(conf), 4),
                    "centroid": (round(cx, 2), round(cy, 2)),
                    "relative_speed_proxy_px_per_frame": round(relative_speed_proxy_px_per_frame, 2)
                }
                tracked_objects.append(tracked_obj)

                # Draw bounding box + Track ID label
                color = self._get_id_color(track_id)
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(annotated_frame, (ix1, iy1), (ix2, iy2), color, 2)
                
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (ix1, iy1 - 20), (ix1 + w, iy1), color, -1)
                cv2.putText(annotated_frame, label, (ix1, iy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Draw trajectory trail
                points = self.track_history[track_id]
                for p_idx in range(1, len(points)):
                    pt1 = (int(points[p_idx-1][0]), int(points[p_idx-1][1]))
                    pt2 = (int(points[p_idx][0]), int(points[p_idx][1]))
                    cv2.line(annotated_frame, pt1, pt2, color, 2)

        return {
            "annotated_frame": annotated_frame,
            "tracked_objects": tracked_objects,
            "active_vehicle_count": len(tracked_objects),
            "total_unique_vehicles_seen": len(self.seen_vehicle_ids),
            "class_counts": class_counts
        }

    def process_video_file(self, video_path: str, output_path: str = "data/processed/test_tracking_output.mp4", max_frames: int = 300) -> Dict[str, Any]:
        """Processes real video with ByteTrack, saving annotated output with IDs and persistent tracking statistics."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Real traffic video file missing at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.reset_tracker()
        frame_count = 0
        active_counts_history = []
        total_detections_count = 0

        start_time = time.time()

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            res = self.track_frame(frame, frame_idx=frame_count)
            ann_frame = res["annotated_frame"]

            active_cnt = res["active_vehicle_count"]
            unique_total = res["total_unique_vehicles_seen"]
            total_detections_count += active_cnt
            active_counts_history.append(active_cnt)

            # Overlay HUD stats
            hud_text = f"Frame: {frame_count}/{max_frames} | Active Tracked: {active_cnt} | Unique IDs Seen: {unique_total}"
            cv2.putText(ann_frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out.write(ann_frame)

        cap.release()
        out.release()

        elapsed_sec = max(0.001, time.time() - start_time)
        processing_fps = round(frame_count / elapsed_sec, 2)
        avg_tracked_per_frame = round(float(np.mean(active_counts_history)) if active_counts_history else 0.0, 2)

        return {
            "video_path": video_path,
            "resolution": f"{width}x{height}",
            "frames_processed": frame_count,
            "processing_time_sec": round(elapsed_sec, 2),
            "processing_fps": processing_fps,
            "total_detections_count": total_detections_count,
            "total_unique_vehicle_ids": len(self.seen_vehicle_ids),
            "average_tracked_vehicles_per_frame": avg_tracked_per_frame,
            "output_path": output_path
        }

    def _get_id_color(self, track_id: int) -> Tuple[int, int, int]:
        np.random.seed(int(track_id) * 17)
        color = np.random.randint(50, 255, size=3).tolist()
        return (int(color[0]), int(color[1]), int(color[2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ByteTrack Real Video Tracking Processor")
    parser.add_argument("--video", type=str, default="data/raw/videos/test_traffic.mp4", help="Path to real traffic video file")
    parser.add_argument("--output", type=str, default="data/processed/test_tracking_output.mp4", help="Path for tracking output video")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames to process")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Real traffic video not found at: {args.video}")
        print("Please place a traffic video at data/raw/videos/test_traffic.mp4")
    else:
        tracker = VehicleTracker()
        stats = tracker.process_video_file(args.video, args.output, max_frames=args.max_frames)
        print("\n=== BYTE TRACK VIDEO TRACKING STATISTICS ===")
        print(stats)
