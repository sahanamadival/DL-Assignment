import numpy as np
from typing import Dict, Any, Optional
from src.config import load_config, CongestionThresholdsConfig

class CongestionEstimator:
    """Calculates project-defined heuristic congestion score [0.0 - 1.0] and level from observed traffic parameters.

    METHODOLOGY NOTE:
    This congestion index is a project-defined empirical heuristic mapping image-space occupancy density
    and observed active vehicle counts. It is not an uncalibrated highway capacity claim.
    """

    def __init__(self, thresholds: Optional[CongestionThresholdsConfig] = None):
        cfg = load_config()
        self.thresholds = thresholds or cfg.traffic_analysis.congestion_thresholds

    def calculate_congestion(
        self,
        density: float,
        vehicle_count: int,
        relative_speed_proxy: float = 0.0,
        max_reference_count: int = 30
    ) -> Dict[str, Any]:
        """Calculates project-defined heuristic congestion score [0.0 - 1.0] and level.

        Formula:
            - Image Density Ratio Weight: 0.50
            - Volume Reference Ratio Weight: 0.50 (vehicle_count / max_reference_count)
        """
        density_norm = float(np.clip(density, 0.0, 1.0))
        volume_ratio = float(np.clip(vehicle_count / max(1, max_reference_count), 0.0, 1.0))

        congestion_score = (0.50 * density_norm) + (0.50 * volume_ratio)
        congestion_score = round(float(np.clip(congestion_score, 0.0, 1.0)), 4)

        if congestion_score <= self.thresholds.low_max:
            level = "LOW"
            color_hex = "#2ecc71"
        elif congestion_score <= self.thresholds.moderate_max:
            level = "MODERATE"
            color_hex = "#f39c12"
        elif congestion_score <= self.thresholds.high_max:
            level = "HIGH"
            color_hex = "#e67e22"
        else:
            level = "SEVERE"
            color_hex = "#e74c3c"

        return {
            "congestion_score": congestion_score,
            "congestion_level": level,
            "color_hex": color_hex,
            "components": {
                "density_factor": round(density_norm, 4),
                "volume_ratio": round(volume_ratio, 4),
                "relative_speed_proxy": round(float(relative_speed_proxy), 2)
            }
        }

if __name__ == "__main__":
    estimator = CongestionEstimator()
    res = estimator.calculate_congestion(density=0.45, vehicle_count=18, relative_speed_proxy=3.5)
    print("Congestion Estimator Test Output:", res)
