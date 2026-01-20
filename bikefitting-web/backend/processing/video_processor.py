"""
Video Processor - Uses Gaussian Process Active Learning for efficient bike fitting.
Based on Fivos's SmartBikeFitter implementation.

Instead of processing ALL frames, uses uncertainty-based acquisition to 
sample only ~30 frames, then fits a GP to predict the full joint angle curves.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False
    print("Warning: BoTorch not available. Falling back to percentile-based estimation.")

from .bike_segmenter import BikeSegmenter
from .angle_predictor import BikeAnglePredictor
from .pose_detector import PoseDetector


class ALSimExperiment:
    """
    Active Learning Experiment using Gaussian Processes.
    Uses uncertainty-based acquisition to select which frames to sample.
    
    This is the core innovation from Fivos's work - instead of processing
    all frames, we use GP variance to intelligently select the most 
    informative frames to sample.
    """
    def __init__(self, timestamps: np.ndarray, kernel_type: str = 'rbf', 
                 acq_strategy: str = 'joint_uncertainty'):
        self.timestamps = timestamps
        self.total_frames = len(timestamps)
        self.y_values = np.full(self.total_frames, np.nan)
        self.visited_indices = []
        self.train_indices = []
        self.wasted_indices = []
        self.kernel_type = kernel_type
        self.acq_strategy = acq_strategy
        self.model = None
        
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        self.X_all = torch.tensor(timestamps.reshape(-1, 1), dtype=torch.double)
        self.X_all_norm = torch.tensor(
            self.x_scaler.fit_transform(timestamps.reshape(-1, 1)),
            dtype=torch.double
        )

    def update_model(self, fit: bool = False):
        """Update the GP model with current observations."""
        if len(self.train_indices) == 0:
            return
        
        X_train = self.timestamps[self.train_indices].reshape(-1, 1)
        Y_train = self.y_values[self.train_indices].reshape(-1, 1)
        
        X_train_norm = torch.tensor(self.x_scaler.transform(X_train), dtype=torch.double)
        Y_train_norm = torch.tensor(self.y_scaler.fit_transform(Y_train), dtype=torch.double)
        
        time_std = self.x_scaler.scale_[0]
        
        old_state_dict = None
        if self.model is not None and not fit:
            old_state_dict = {
                k: v for k, v in self.model.state_dict().items()
                if 'train_inputs' not in k and 'train_targets' not in k
            }
        
        # Define kernel with appropriate priors
        fast_ls_target = 0.40 / time_std
        fast_ls_prior = GammaPrior(concentration=2.0, rate=2.0/fast_ls_target)
        
        if self.kernel_type == 'rbf':
            covar = ScaleKernel(RBFKernel(lengthscale_prior=fast_ls_prior))
        else:
            covar = ScaleKernel(MaternKernel(nu=2.5, lengthscale_prior=fast_ls_prior))
        
        self.model = SingleTaskGP(X_train_norm, Y_train_norm, covar_module=covar)
        
        if fit:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass
        elif old_state_dict is not None:
            self.model.load_state_dict(old_state_dict, strict=False)

    def _get_model_variance(self, candidate_indices: np.ndarray) -> torch.Tensor:
        """Get posterior variance at candidate points."""
        if self.model is None:
            return torch.ones(len(candidate_indices))
        X_candidates = self.X_all_norm[candidate_indices]
        self.model.eval()
        with torch.no_grad():
            return self.model(X_candidates).variance

    def select_next_point(self, other_experiments: list = None) -> Optional[int]:
        """
        Select next point to sample using acquisition function.
        Uses joint uncertainty across multiple experiments (e.g., knee + hip).
        """
        visited_mask = np.zeros(self.total_frames, dtype=bool)
        visited_mask[self.visited_indices] = True
        candidate_indices = np.where(~visited_mask)[0]
        
        if len(candidate_indices) == 0:
            return None
        
        if self.acq_strategy == 'random':
            return np.random.choice(candidate_indices)
        
        my_variance = self._get_model_variance(candidate_indices)
        
        if self.acq_strategy == 'joint_uncertainty' and other_experiments:
            variances = [my_variance]
            for exp in other_experiments:
                variances.append(exp._get_model_variance(candidate_indices))
            stacked = torch.stack(variances)
            
            # Weighted: 80% knee, 20% hip (knee is most important for saddle height)
            if stacked.shape[0] == 2:
                weights = torch.tensor([0.8, 0.2], dtype=torch.double)
                final_variance = torch.sum(stacked * weights.view(-1, 1), dim=0)
            else:
                final_variance, _ = torch.max(stacked, dim=0)
        else:
            final_variance = my_variance
        
        # Spatial suppression: avoid sampling near failed detections
        if len(self.wasted_indices) > 0:
            candidate_times = torch.tensor(self.timestamps[candidate_indices], dtype=torch.float32)
            wasted_times = torch.tensor(self.timestamps[self.wasted_indices], dtype=torch.float32)
            dists = torch.abs(candidate_times.unsqueeze(1) - wasted_times.unsqueeze(0))
            min_dists, _ = torch.min(dists, dim=1)
            too_close = min_dists < 1.0  # 1 second suppression radius
            final_variance[too_close] = -1.0
            if torch.max(final_variance) == -1.0:
                final_variance[too_close] = 0.0
        
        return candidate_indices[torch.argmax(final_variance).item()]

    def add_observation(self, idx: int, value: float, fit: bool = False) -> bool:
        """Add an observation and update the model."""
        if idx in self.visited_indices:
            return False
        
        self.visited_indices.append(idx)
        
        if np.isnan(value):
            self.wasted_indices.append(idx)
            return False
        
        self.train_indices.append(idx)
        self.y_values[idx] = value
        self.update_model(fit=fit)
        return True

    def predict_curve(self) -> Optional[np.ndarray]:
        """Predict the full curve using the fitted GP."""
        if self.model is None:
            return None
        self.model.eval()
        with torch.no_grad():
            pred_y_norm = self.model(self.X_all_norm).mean
        return self.y_scaler.inverse_transform(pred_y_norm.numpy().reshape(-1, 1)).flatten()


def generate_bike_fit_recommendations(results: Dict) -> Dict:
    """
    Generate bike fit recommendations from GP-predicted metrics.
    Based on the Holmes method and standard bike fitting heuristics.
    """
    k_max = results.get('max_knee_ext', 0) or results.get('knee_max_extension', 0)
    k_min = results.get('min_knee_flex', 70) or results.get('knee_min_flexion', 70)
    h_min = results.get('min_hip_angle', 60)
    e_avg = results.get('avg_elbow_angle', 155)
    
    recommendations = {
        "saddle_height": {"status": "ok", "action": None, "adjustment_mm": 0, "details": ""},
        "saddle_fore_aft": {"status": "ok", "action": None, "adjustment_mm": 0, "details": ""},
        "crank_length": {"status": "ok", "action": None, "details": ""},
        "cockpit": {"status": "ok", "reach_action": None, "adjustment_mm": 0, "details": ""},
        "summary": [],
        "metrics": {
            "knee_max_extension": round(k_max, 1) if k_max else None,
            "knee_min_flexion": round(k_min, 1) if k_min else None,
            "min_hip_angle": round(h_min, 1) if h_min else None,
            "avg_elbow_angle": round(e_avg, 1) if e_avg else None
        }
    }
    
    if not k_max or k_max == 0:
        recommendations["summary"].append("Not enough data for recommendations")
        return recommendations
    
    # 1. SADDLE HEIGHT (target: 140-150° knee extension at bottom of stroke)
    k_target_min, k_target_max = 140.0, 150.0
    
    if k_max < k_target_min:
        deg_diff = k_target_min - k_max
        mm_adj = deg_diff * 2.0  # ~2mm per degree heuristic
        recommendations["saddle_height"] = {
            "status": "low", "action": "raise", "adjustment_mm": round(mm_adj),
            "details": f"Knee extension {k_max:.0f}° below optimal ({k_target_min}-{k_target_max}°). Raise saddle ~{mm_adj:.0f}mm."
        }
        recommendations["summary"].append(f"Raise saddle ~{mm_adj:.0f}mm")
    elif k_max > k_target_max:
        deg_diff = k_max - k_target_max
        mm_adj = deg_diff * 2.0
        recommendations["saddle_height"] = {
            "status": "high", "action": "lower", "adjustment_mm": round(mm_adj),
            "details": f"Knee extension {k_max:.0f}° indicates overextension risk. Lower saddle ~{mm_adj:.0f}mm."
        }
        recommendations["summary"].append(f"Lower saddle ~{mm_adj:.0f}mm")
    else:
        recommendations["saddle_height"]["details"] = f"Knee extension {k_max:.0f}° is within optimal range."
        recommendations["summary"].append("Saddle height is optimal")
    
    # 2. SADDLE FORE/AFT (knee should be >70° at top of stroke)
    if k_min and k_min < 70.0:
        recommendations["saddle_fore_aft"] = {
            "status": "forward", "action": "move_back", "adjustment_mm": 10,
            "details": f"Knee angle at top of stroke ({k_min:.0f}°) is too closed. Move saddle back 5-10mm."
        }
        if recommendations["saddle_height"]["adjustment_mm"] < 10:
            recommendations["summary"].append("Move saddle back 5-10mm")
    
    # 3. CRANK LENGTH (check for hip impingement)
    if h_min and (h_min < 48.0 or (k_min and k_min < 68.0)):
        recommendations["crank_length"] = {
            "status": "issue", "action": "consider_shorter",
            "details": f"Hip angle {h_min:.0f}° indicates impingement at top of stroke. Consider shorter cranks (-5mm)."
        }
        recommendations["summary"].append("Consider shorter cranks")
    
    # 4. COCKPIT (elbow should be 150-165°)
    if e_avg and e_avg > 165.0:
        mm_reduce = max(10, ((e_avg - 160.0) / 5.0) * 10)
        recommendations["cockpit"] = {
            "status": "issue", "reach_action": "shorten", "adjustment_mm": round(mm_reduce),
            "details": f"Arms locked out ({e_avg:.0f}°). Shorten stem ~{mm_reduce:.0f}mm."
        }
        recommendations["summary"].append(f"Shorten stem ~{mm_reduce:.0f}mm")
    elif e_avg and e_avg < 150.0:
        mm_extend = max(10, ((150.0 - e_avg) / 5.0) * 10)
        recommendations["cockpit"] = {
            "status": "issue", "reach_action": "lengthen", "adjustment_mm": round(mm_extend),
            "details": f"Arms very bent ({e_avg:.0f}°). Lengthen stem ~{mm_extend:.0f}mm."
        }
        recommendations["summary"].append(f"Lengthen stem ~{mm_extend:.0f}mm")
    
    return recommendations


def apply_perspective_correction(angle_2d: float, bike_yaw: float) -> float:
    """
    Correct 2D joint angle for perspective distortion based on bike yaw.
    When bike is not exactly perpendicular (90°), 2D angles appear compressed.
    """
    deviation_deg = abs(abs(bike_yaw) - 90.0)
    deviation_deg = min(deviation_deg, 30.0)  # Cap correction
    correction = 1.0 / np.cos(np.radians(deviation_deg))
    return angle_2d * correction


class VideoProcessor:
    """
    Smart Bike Fitter using Gaussian Process Active Learning.
    
    Key innovation: Instead of processing ALL frames, uses uncertainty-based
    acquisition to sample only ~30 frames, then fits a GP to predict the 
    full joint angle curves.
    
    This provides:
    1. Much faster processing (30 pose detections vs hundreds)
    2. Better estimates via GP interpolation
    3. Uses concepts from MLPW course (BO, GP, acquisition functions)
    """
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing VideoProcessor on {self.device}")
        print("Loading bike segmenter (YOLOv8n-seg)...")
        self.segmenter = BikeSegmenter()
        
        print("Loading bike angle predictor (ConvNeXt)...")
        self.angle_predictor = BikeAnglePredictor(model_path, self.device)
        
        print("Loading pose detector (YOLOv8m-pose)...")
        self.pose_detector = PoseDetector()
        
        print(f"Models loaded! BoTorch available: {HAS_BOTORCH}")
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        csv_path: Optional[str] = None,
        output_fps: int = 10,
        max_duration_sec: Optional[float] = None,
        start_time: float = 0,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        n_samples: int = 30
    ) -> Dict:
        """
        Process a video using GP-based Active Learning.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            n_samples: Number of frames to sample with AL (default 30)
            
        Returns:
            Dict with processing stats, frame_data, and recommendations
        """
        cap = cv2.VideoCapture(input_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        start_frame = int(start_time * video_fps) if start_time > 0 else 0
        if end_time and end_time > start_time:
            end_frame = min(int(end_time * video_fps), total_frames)
        else:
            end_frame = total_frames
        if max_duration_sec:
            max_frames = int(max_duration_sec * video_fps)
            end_frame = min(end_frame, start_frame + max_frames)
        
        frames_in_range = end_frame - start_frame
        frame_skip = max(1, int(video_fps / 30))  # Scan at 30fps
        frames_to_scan = frames_in_range // frame_skip
        
        print(f"Phase 1: Scanning {frames_to_scan} frames for valid side-view angles...")
        
        # Output video setup
        main_height = min(720, orig_height)
        main_width = int(orig_width * (main_height / orig_height))
        if main_width > 1280:
            main_width, main_height = 1280, int(orig_height * (1280 / orig_width))
        scale = main_height / orig_height
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (main_width, main_height))
        
        # Phase 1: Scan for valid side-view frames
        valid_frames = []
        bike_angles_all = []
        
        for i in tqdm(range(frames_to_scan), desc="Scanning"):
            frame_num = start_frame + (i * frame_skip)
            if frame_num >= total_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            masked, mask, success = self.segmenter.mask_bike(frame)
            if not success:
                continue
            
            yaw, confidence = self.angle_predictor.predict(masked)
            bike_angles_all.append(yaw)
            
            # Gating: only side-view frames (60-120 degrees absolute)
            if 60 <= abs(yaw) <= 120:
                physical_time = frame_num / video_fps
                valid_frames.append({
                    'idx': frame_num,
                    'time': physical_time,
                    'yaw': yaw,
                    'frame': frame.copy(),
                    'mask': mask
                })
            
            if progress_callback:
                progress_callback(i + 1, frames_to_scan + n_samples)
        
        if len(valid_frames) < 10:
            print("Error: Not enough valid side-view frames found.")
            cap.release()
            out.release()
            return {
                "stats": {"frames_processed": 0, "error": "Not enough side-view frames"},
                "frame_data": []
            }
        
        print(f"Found {len(valid_frames)} valid frames.")
        
        # Phase 2: Active Learning with GP (if BoTorch available)
        if HAS_BOTORCH and len(valid_frames) >= n_samples:
            print(f"Phase 2: Active Learning with GP ({n_samples} samples)...")
            results, frame_data = self._process_with_gp(
                cap, valid_frames, n_samples, progress_callback, 
                frames_to_scan, out, main_width, main_height, scale
            )
            method = "Gaussian Process Active Learning"
        else:
            print("Phase 2: Fallback to percentile-based estimation...")
            results, frame_data = self._process_fallback(
                valid_frames, out, main_width, main_height, scale
            )
            method = "Percentile-based estimation"
        
        cap.release()
        out.release()
        
        # Generate recommendations
        recommendations = generate_bike_fit_recommendations(results)
        
        stats = {
            "frames_processed": len(valid_frames),
            "output_fps": output_fps,
            "valid_frames": len(valid_frames),
            "knee_max_extension": results.get('max_knee_ext', 0),
            "knee_min_flexion": results.get('min_knee_flex', 70),
            "min_hip_angle": results.get('min_hip_angle', 60),
            "avg_elbow_angle": results.get('avg_elbow_angle', 155),
            "recommendations": recommendations,
            "method": method
        }
        
        if bike_angles_all:
            stats["avg_bike_angle"] = float(np.mean(bike_angles_all))
        
        print(f"\nProcessing complete using {method}!")
        print(f"Output saved to: {output_path}")
        
        return {"stats": stats, "frame_data": frame_data}
    
    def _process_with_gp(self, cap, valid_frames, n_samples, progress_callback,
                         frames_scanned, out, out_w, out_h, scale):
        """Process using GP Active Learning."""
        timestamps = np.array([f['time'] for f in valid_frames])
        
        # Create GP experiments for each joint
        exp_knee = ALSimExperiment(timestamps, kernel_type='rbf', acq_strategy='joint_uncertainty')
        exp_hip = ALSimExperiment(timestamps, kernel_type='rbf', acq_strategy='joint_uncertainty')
        exp_elbow = ALSimExperiment(timestamps, kernel_type='rbf', acq_strategy='joint_uncertainty')
        
        # Initialize with 5 random samples
        n_init = min(5, len(valid_frames))
        init_indices = np.random.choice(len(valid_frames), n_init, replace=False)
        self.pose_detector.reset_smoother()
        
        for local_idx in init_indices:
            self._process_sample_gp(valid_frames[local_idx], local_idx, 
                                   exp_knee, exp_hip, exp_elbow, fit=False)
        
        exp_knee.update_model(fit=True)
        exp_hip.update_model(fit=True)
        exp_elbow.update_model(fit=True)
        
        # Active learning loop
        samples_taken = n_init
        pbar = tqdm(total=n_samples, initial=n_init, desc="Active Learning")
        
        while samples_taken < n_samples:
            next_idx = exp_knee.select_next_point(other_experiments=[exp_hip])
            if next_idx is None:
                break
            
            do_fit = ((samples_taken + 1) % 5 == 0)
            self._process_sample_gp(valid_frames[next_idx], next_idx,
                                   exp_knee, exp_hip, exp_elbow, fit=do_fit)
            samples_taken += 1
            pbar.update(1)
            
            if progress_callback:
                progress_callback(frames_scanned + samples_taken, frames_scanned + n_samples)
        
        pbar.close()
        
        # Predict full curves
        pred_knee = exp_knee.predict_curve()
        pred_hip = exp_hip.predict_curve()
        pred_elbow = exp_elbow.predict_curve()
        
        results = {
            'max_knee_ext': float(np.max(pred_knee)) if pred_knee is not None else np.nan,
            'min_knee_flex': float(np.min(pred_knee)) if pred_knee is not None else np.nan,
            'min_hip_angle': float(np.min(pred_hip)) if pred_hip is not None else np.nan,
            'avg_elbow_angle': float(np.mean(pred_elbow)) if pred_elbow is not None else np.nan
        }
        
        # Generate output video
        frame_data = []
        for i, vf in enumerate(valid_frames[:min(100, len(valid_frames))]):
            output = cv2.resize(vf['frame'], (out_w, out_h))
            pose = self.pose_detector.detect(vf['frame'])
            
            if pose["keypoints_xy"] is not None and pose.get("detected_side"):
                output = self.pose_detector.draw_skeleton(
                    output, pose["keypoints_xy"], pose["keypoints_conf"],
                    pose["detected_side"], scale
                )
            
            if vf['mask'] is not None and vf['mask'].max() > 0:
                mask_resized = cv2.resize(vf['mask'], (out_w, out_h))
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, 
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            
            out.write(output)
            
            frame_data.append({
                "frame": i,
                "time": vf['time'],
                "bike_angle": float(vf['yaw']),
                "knee_angle": float(pred_knee[i]) if pred_knee is not None and i < len(pred_knee) else None,
                "hip_angle": float(pred_hip[i]) if pred_hip is not None and i < len(pred_hip) else None,
                "elbow_angle": float(pred_elbow[i]) if pred_elbow is not None and i < len(pred_elbow) else None,
                "is_valid": True
            })
        
        return results, frame_data
    
    def _process_sample_gp(self, frame_info, idx, exp_knee, exp_hip, exp_elbow, fit):
        """Process a single frame and update GP experiments."""
        self.pose_detector.reset_smoother()
        pose = self.pose_detector.detect(frame_info['frame'])
        angles = pose.get('angles', {})
        
        # Apply perspective correction
        yaw = frame_info['yaw']
        k = angles.get('knee_angle', np.nan)
        h = angles.get('hip_angle', np.nan)
        e = angles.get('elbow_angle', np.nan)
        
        if not np.isnan(k):
            k = apply_perspective_correction(k, yaw)
        if not np.isnan(h):
            h = apply_perspective_correction(h, yaw)
        if not np.isnan(e):
            e = apply_perspective_correction(e, yaw)
        
        exp_knee.add_observation(idx, k, fit=fit)
        exp_hip.add_observation(idx, h, fit=fit)
        exp_elbow.add_observation(idx, e, fit=fit)
    
    def _process_fallback(self, valid_frames, out, out_w, out_h, scale):
        """Fallback: process all frames with percentile estimation."""
        knee_angles = []
        hip_angles = []
        elbow_angles = []
        frame_data = []
        
        self.pose_detector.reset_smoother()
        
        for i, vf in enumerate(tqdm(valid_frames, desc="Processing")):
            pose = self.pose_detector.detect(vf['frame'])
            angles = pose.get('angles', {})
            
            yaw = vf['yaw']
            k = angles.get('knee_angle')
            h = angles.get('hip_angle')
            e = angles.get('elbow_angle')
            
            if k is not None:
                k = apply_perspective_correction(k, yaw)
                knee_angles.append(k)
            if h is not None:
                h = apply_perspective_correction(h, yaw)
                hip_angles.append(h)
            if e is not None:
                e = apply_perspective_correction(e, yaw)
                elbow_angles.append(e)
            
            # Write output frame
            output = cv2.resize(vf['frame'], (out_w, out_h))
            if pose["keypoints_xy"] is not None and pose.get("detected_side"):
                output = self.pose_detector.draw_skeleton(
                    output, pose["keypoints_xy"], pose["keypoints_conf"],
                    pose["detected_side"], scale
                )
            if vf['mask'] is not None and vf['mask'].max() > 0:
                mask_resized = cv2.resize(vf['mask'], (out_w, out_h))
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            out.write(output)
            
            frame_data.append({
                "frame": i,
                "time": vf['time'],
                "bike_angle": float(yaw),
                "knee_angle": float(k) if k else None,
                "hip_angle": float(h) if h else None,
                "elbow_angle": float(e) if e else None,
                "is_valid": True
            })
        
        results = {
            'max_knee_ext': float(np.percentile(knee_angles, 95)) if knee_angles else np.nan,
            'min_knee_flex': float(np.percentile(knee_angles, 5)) if knee_angles else np.nan,
            'min_hip_angle': float(np.percentile(hip_angles, 5)) if hip_angles else np.nan,
            'avg_elbow_angle': float(np.mean(elbow_angles)) if elbow_angles else np.nan
        }
        
        return results, frame_data
