"""
Smart Bike Fitter - Uses Bayesian Optimization to estimate bike fit metrics from video.
Refactored to match ALSimExperiment implementation logic.
"""

import cv2
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, PeriodicKernel
from gpytorch.priors import NormalPrior, GammaPrior
from sklearn.preprocessing import StandardScaler

# Import custom modules
from bike_segmenter import BikeSegmenter
from angle_predictor import BikeAnglePredictor
from pose_detector import PoseDetector

class ALSimExperiment:
    def __init__(self, timestamps, kernel_type='matern', acq_strategy='uncertainty'):
        """
        Adapted ALSimExperiment for live video processing.
        Unlike the benchmark, we don't provide y_values upfront. 
        """
        self.timestamps = timestamps
        self.total_frames = len(timestamps)
        
        # In live mode, y_values are unknown (NaN) until observed
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
        # Pre-fit X scaler on the known timestamps
        self.X_all_norm = torch.tensor(
            self.x_scaler.fit_transform(timestamps.reshape(-1, 1)), 
            dtype=torch.double
        )

    def update_model(self, fit=False):
        if len(self.train_indices) == 0:
            return

        X_train = self.timestamps[self.train_indices].reshape(-1, 1)
        Y_train = self.y_values[self.train_indices].reshape(-1, 1)
        
        # Transform data
        X_train_norm = torch.tensor(self.x_scaler.transform(X_train), dtype=torch.double)
        Y_train_norm = torch.tensor(self.y_scaler.fit_transform(Y_train), dtype=torch.double)
        
        time_std = self.x_scaler.scale_[0] 

        old_state_dict = None
        if self.model is not None and not fit:
            # We save the learned parameters (lengthscale, noise, etc)
            old_state_dict = self.model.state_dict()
            # Filter out 'train inputs' because the data size is changing
            old_state_dict = {
                k: v for k, v in old_state_dict.items() 
                if 'train_inputs' not in k and 'train_targets' not in k
            }

        # define priors        
        fit_period_mean = 1.10 / time_std
        fit_period_std = 0.30 / time_std
        per_prior = NormalPrior(loc=fit_period_mean, scale=fit_period_std)

        fast_ls_target = 0.40 / time_std
        fast_rate = 2.0 / fast_ls_target
        fast_ls_prior = GammaPrior(concentration=2.0, rate=fast_rate)

        slow_ls_target = 10.0 / time_std
        slow_rate = 2.0 / slow_ls_target
        slow_ls_prior = GammaPrior(concentration=2.0, rate=slow_rate)

        smooth_target = 1.2 
        smooth_prior = GammaPrior(concentration=4.0, rate=4.0/smooth_target)

        # Kernel Selection
        if self.kernel_type == 'rbf':
            covar = ScaleKernel(RBFKernel(lengthscale_prior=fast_ls_prior))
            
        elif self.kernel_type == 'locally_periodic':
            per_kernel = PeriodicKernel(period_length_prior=per_prior, lengthscale_prior=smooth_prior)
            # Initialize with sensible defaults to help convergence
            per_kernel.period_length = fit_period_mean
            per_kernel.lengthscale = smooth_target
            mat_kernel = MaternKernel(nu=2.5, lengthscale_prior=slow_ls_prior)
            covar = ScaleKernel(per_kernel * mat_kernel)
            
        else: # Default to Matern
            covar = ScaleKernel(MaternKernel(nu=2.5, lengthscale_prior=fast_ls_prior))
            
        # Create Model
        self.model = SingleTaskGP(X_train_norm, Y_train_norm, covar_module=covar)

        # Optimization/Loading of kernel params
        if fit:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            try:
                fit_gpytorch_mll(mll)
            except:
                pass
        elif old_state_dict is not None:
            self.model.load_state_dict(old_state_dict, strict=False)

    def _get_model_variance(self, candidate_indices):
        if self.model is None:
            return torch.ones(len(candidate_indices))
        X_candidates = self.X_all_norm[candidate_indices]
        self.model.eval()
        with torch.no_grad():
            posterior = self.model(X_candidates)
            return posterior.variance

    def select_next_point(self, other_experiments=None):
        visited_mask = np.zeros(self.total_frames, dtype=bool)
        visited_mask[self.visited_indices] = True
        candidate_indices = np.where(~visited_mask)[0]
        
        if len(candidate_indices) == 0: return None
        
        if self.acq_strategy == 'random':
            return np.random.choice(candidate_indices)
        
        my_variance = self._get_model_variance(candidate_indices)
        
        if self.acq_strategy == 'joint_uncertainty' and other_experiments is not None:
            variances = [my_variance]
            for exp in other_experiments:
                variances.append(exp._get_model_variance(candidate_indices))
            stacked = torch.stack(variances)
            
            # Weighted Strategy: 0.8 * Knee + 0.2 * Hip
            if stacked.shape[0] == 2:
                weights = torch.tensor([0.8, 0.2], dtype=torch.double)
                weighted_variances = stacked * weights.view(-1, 1)
                final_variance = torch.sum(weighted_variances, dim=0)
            else:
                final_variance, _ = torch.max(stacked, dim=0)
        else:
            final_variance = my_variance

        # Wasted index exclusion logic (Spatial Suppression)
        if len(self.wasted_indices) > 0:
            candidate_times = torch.tensor(self.timestamps[candidate_indices], dtype=torch.float32)
            wasted_times = torch.tensor(self.timestamps[self.wasted_indices], dtype=torch.float32)
            dists = torch.abs(candidate_times.unsqueeze(1) - wasted_times.unsqueeze(0))
            min_dists, _ = torch.min(dists, dim=1)
            suppression_radius = 1.0 
            too_close_mask = min_dists < suppression_radius
            
            # Apply suppression
            final_variance[too_close_mask] = -1.0
            if torch.max(final_variance) == -1.0:
                 final_variance[too_close_mask] = 0.0

        best_local_idx = torch.argmax(final_variance).item()
        return candidate_indices[best_local_idx]

    def add_observation(self, idx, value, fit=False):
        if idx in self.visited_indices: return False
        
        self.visited_indices.append(idx)
        
        # Check if the detected value is valid
        if np.isnan(value):
            self.wasted_indices.append(idx)
            self.y_values[idx] = np.nan # Explicitly mark as NaN
            return False
        else:
            self.train_indices.append(idx)
            self.y_values[idx] = value
            self.update_model(fit=fit)
            return True

    def predict_curve(self):
        if self.model is None: return None
        self.model.eval()
        with torch.no_grad():
            posterior = self.model(self.X_all_norm)
            pred_y_norm = posterior.mean
        return self.y_scaler.inverse_transform(pred_y_norm.numpy().reshape(-1, 1)).flatten()


# Video Processor
class SmartBikeFitter:
    def __init__(self, bike_model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing SmartBikeFitter on {self.device}...")
        
        self.segmenter = BikeSegmenter()
        self.angle_predictor = BikeAnglePredictor(bike_model_path, self.device)
        self.pose_detector = PoseDetector() # Loads YOLO-Pose
        
    def analyze_video(self, video_path, n_samples=30, target_scan_fps=30, start_time=0, end_time=None, max_duration_sec=None):
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
         
        frame_skip = max(1, int(video_fps / target_scan_fps))
        frames_to_process = frames_in_range // frame_skip
        
        print(f"Processing range: {start_time:.1f}s to {(end_frame/video_fps):.1f}s")
        print(f"Phase 1: Scanning {frames_to_process} frames (skip={frame_skip}) for valid bike angles...")
        
        valid_frames = [] # List of {'idx': int, 'time': float}
        
        for i in tqdm(range(frames_to_process), desc="Scanning"):
            frame_num = start_frame + (i * frame_skip)
            if frame_num >= total_frames: break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret: break
            
            # A. Mask Bike
            masked, _, success = self.segmenter.mask_bike(frame)
            if not success: continue
            
            # B. Predict Angle
            yaw, _ = self.angle_predictor.predict(masked)
            
            # C. Gating Logic (60-120 degrees absolute)
            if 60 <= abs(yaw) <= 120:
                physical_time = frame_num / video_fps
                valid_frames.append({'idx': frame_num, 'time': physical_time})

        if len(valid_frames) < 10:
            print("Error: Not enough valid side-view frames found in the specified range.")
            cap.release()
            return None

        print(f"Found {len(valid_frames)} valid frames. Starting Active Learning...")
        
        # Active Learning Setup
        timestamps = np.array([x['time'] for x in valid_frames])
        
        config = [
            {'name': 'Joint Uncertainty (RBF)', 'kernel': 'rbf', 'acq': 'joint_uncertainty'}
        ]
        exp_knee = ALSimExperiment(timestamps, kernel_type=config[0]['kernel'], acq_strategy=config[0]['acq'])
        exp_hip = ALSimExperiment(timestamps, kernel_type=config[0]['kernel'], acq_strategy=config[0]['acq'])
        exp_elbow = ALSimExperiment(timestamps, kernel_type=config[0]['kernel'], acq_strategy=config[0]['acq'])

        # A. Initialization (5 random points)
        init_indices = np.random.choice(len(valid_frames), 5, replace=False)
        self.pose_detector.reset_smoother()
        
        for local_idx in tqdm(init_indices, desc="Initializing"):
            self._process_sample(cap, valid_frames[local_idx], local_idx, exp_knee, exp_hip, exp_elbow, fit=False)
        
        exp_knee.update_model(fit=True)
        exp_hip.update_model(fit=True)
        exp_elbow.update_model(fit=True)
            
        # B. Optimization Loop
        samples_taken = 5
        pbar = tqdm(total=n_samples, initial=5, desc="Optimizing")
        
        while samples_taken < n_samples:
            # Ask Knee Experiment for next best point, passing Hip experiment for joint uncertainty
            next_idx = exp_knee.select_next_point(other_experiments=[exp_hip])
            
            if next_idx is None: break
            
            # Optimize hyperparameters every 5 samples
            do_fit = ((samples_taken + 1) % 5 == 0)
            
            self._process_sample(cap, valid_frames[next_idx], next_idx, exp_knee, exp_hip, exp_elbow, fit=do_fit)
            
            samples_taken += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        # Predict results
        results = self._predict_results(exp_knee, exp_hip, exp_elbow)
        self._generate_detailed_recommendations(results)
        return results

    def _generate_detailed_recommendations(self, res):
        """
        Generates specific adjustment recommendations in mm based on biomechanical heuristics.
        Order of operations: Saddle Height -> Saddle Fore/Aft -> Crank Length -> Cockpit
        """
        print("\n" + "="*50)
        print("          BIKE FIT RECOMMENDATION ENGINE          ")
        print("="*50)
        
        # --- 1. SADDLE HEIGHT (The Foundation) ---
        k_max = res['max_knee_ext']
        k_target_min, k_target_max = 140.0, 150.0
        saddle_height_adj = 0.0
        
        print(f"\n[1] SADDLE HEIGHT (Knee Ext: {k_max:.1f}° | Target: {k_target_min}-{k_target_max}°)")
        
        if k_max < k_target_min:
            deg_diff = k_target_min - k_max
            mm_adj = deg_diff * 2.0
            saddle_height_adj = mm_adj
            print(f"  ❌ STATUS: Saddle too LOW.")
            print(f"  🔧 ACTION: RAISE saddle by approx {mm_adj:.0f} mm.")
            
        elif k_max > k_target_max:
            deg_diff = k_max - k_target_max
            mm_adj = deg_diff * 2.0
            saddle_height_adj = -mm_adj
            print(f"  ❌ STATUS: Saddle too HIGH (Risk of overextension).")
            print(f"  🔧 ACTION: LOWER saddle by approx {mm_adj:.0f} mm.")
        else:
            print(f"  ✅ STATUS: Saddle height is within optimal range.")

        # --- 2. SADDLE FORE/AFT ---
        k_min = res['min_knee_flex'] # Knee angle at top of stroke
        print(f"\n[2] SADDLE FORE/AFT (Min Knee Flex: {k_min:.1f}° | Target: >70°)")
        
        fore_aft_issue = False
        if k_min < 70.0:
            print(f"  ⚠️ STATUS: Knee too closed at top of stroke (<70°).")
            fore_aft_issue = True
            if saddle_height_adj > 10:
                 print(f"  ℹ️ NOTE: Raising the saddle (Step 1) will naturally open this angle.")
            else:
                 print(f"  🔧 ACTION: Move saddle BACKWARD by 5-10 mm.")
        else:
            print(f"  ✅ STATUS: Clearance at top of stroke is good.")

        # --- 3. CRANK LENGTH (Hardware Check) ---
        # This solves issues where Hip/Knee are compressed despite good saddle height.
        h_min = res['min_hip_angle']
        print(f"\n[3] CRANK LENGTH (Hip Clearance: {h_min:.1f}° | Target: >45-50°)")
        
        # Heuristic: If hip is pinched (<48) OR knee is very tight (<68)
        if h_min < 48.0 or k_min < 68.0:
            print(f"  ❌ STATUS: CRITICAL IMPINGEMENT detected at top of stroke.")
            print(f"     (Hip: {h_min:.1f}° or Knee: {k_min:.1f}° are too acute)")
            print(f"  🔧 ACTION: Considerations for SHORTER CRANKS (e.g., -5mm).")
            print(f"     Why? Shortening cranks opens the hip angle significantly.")
            print(f"     ⚠️ NOTE: If you shorten cranks by 5mm, you must RAISE saddle by 5mm.")
        else:
            print(f"  ✅ STATUS: Crank length appears appropriate (No impingement).")

        # --- 4. COCKPIT (Reach & Stack) ---
        e_avg = res['avg_elbow_angle']
        print(f"\n[4] COCKPIT ADJUSTMENT (Reach & Stack)")
        print(f"    Elbow Avg: {e_avg:.1f}° (Target: 150-160°)")
        
        # Stack (Handlebar Height) - Driven by Hip Angle
        stack_change = 0
        if h_min < 50.0:
             print(f"  ❌ STACK: Hip angle closed. If cranks are okay, RAISE handlebars.")
             stack_change = 1 
        elif h_min > 65.0 and e_avg > 160:
             print(f"  ℹ️ STACK: Hip angle open. Potential for lower aero position.")

        # Reach (Stem Length) - Driven by Elbow Angle
        if e_avg > 165.0:
            print(f"  ❌ REACH: Arms locked out (>165°).")
            deg_over = e_avg - 160.0
            mm_reduce = max(10, (deg_over / 5.0) * 10)
            print(f"  🔧 ACTION: SHORTEN stem by {mm_reduce:.0f} mm.")
                
        elif e_avg < 150.0:
            print(f"  ⚠️ REACH: Arms very bent (<150°).")
            deg_under = 150.0 - e_avg
            mm_extend = max(10, (deg_under / 5.0) * 10)
            print(f"  🔧 ACTION: LENGTHEN stem by {mm_extend:.0f} mm.")
        else:
            print(f"  ✅ REACH: Good elbow bend.")

        print("="*50 + "\n")
        
    def _process_sample(self, cap, frame_info, idx, exp_knee, exp_hip, exp_elbow, fit):
        """Helper to seek video, detect pose, and update experiments."""
        real_frame_idx = frame_info['idx']
        self.pose_detector.reset_smoother() # critical
        cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_idx)
        ret, frame = cap.read()
        
        # Values default to NaN if detection fails
        k, h, e = np.nan, np.nan, np.nan
        
        if ret:
            # Run Pose Detection
            pose = self.pose_detector.detect(frame)
            angles = pose.get('angles', {})
            
            k = angles.get('knee_angle', np.nan)
            h = angles.get('hip_angle', np.nan)
            e = angles.get('elbow_angle', np.nan)
        
        # Update all experiments
        # We pass the specific value to each experiment
        exp_knee.add_observation(idx, k, fit=fit)
        exp_hip.add_observation(idx, h, fit=fit)
        exp_elbow.add_observation(idx, e, fit=fit)

    def _predict_results(self, exp_knee, exp_hip, exp_elbow):
        # Knee Prediction
        pred_knee = exp_knee.predict_curve()
        max_knee = np.max(pred_knee) if pred_knee is not None else np.nan
        min_knee = np.min(pred_knee) if pred_knee is not None else np.nan
        
        # Hip Prediction
        pred_hip = exp_hip.predict_curve()
        min_hip = np.min(pred_hip) if pred_hip is not None else np.nan
        
        # Elbow Prediction (Simple Average of observed values is often sufficient, 
        # but using GP mean handles noise better if we have it)
        pred_elbow = exp_elbow.predict_curve()
        avg_elbow = np.mean(pred_elbow) if pred_elbow is not None else np.nan

        return {
            'max_knee_ext': max_knee,
            'min_knee_flex': min_knee,
            'min_hip_angle': min_hip,
            'avg_elbow_angle': avg_elbow
        }



# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Bike Fitter using Bayesian Optimization")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--model", default="../../../models/best_model.pt", help="Path to the bike angle model")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to optimize (default: 30)")
    
    # Time window arguments
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds (0 for end of file)")
    parser.add_argument("--duration", type=float, default=0, help="Max duration in seconds")
    
    args = parser.parse_args()
    
    # Handle zeros as None
    end_val = args.end if args.end > 0 else None
    dur_val = args.duration if args.duration > 0 else None
    
    if os.path.exists(args.video) and os.path.exists(args.model):
        fitter = SmartBikeFitter(args.model)
        fitter.analyze_video(
            args.video, 
            n_samples=args.samples,
            start_time=args.start,
            end_time=end_val,
            max_duration_sec=dur_val
        )
    else:
        print(f"Error: File not found.")