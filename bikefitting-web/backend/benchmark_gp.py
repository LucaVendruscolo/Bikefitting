import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, PeriodicKernel
from gpytorch.priors import NormalPrior, GammaPrior
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==========================================
# 1. DYNAMIC DATA LOADER
# ==========================================
def load_real_data_dynamic(csv_filename):
    """
    Loads real cyclist data and calculates robust Ground Truth metrics 
    specific to THAT video file (using 98th/2nd percentiles).
    """
    if not os.path.exists(csv_filename):
        return None, None, None

    try:
        df = pd.read_csv(csv_filename)
    except:
        return None, None, None
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    timestamps = df['time'].values
    
    # Extract columns
    knee = df['knee_angle'].values if 'knee_angle' in df.columns else np.full_like(timestamps, np.nan)
    hip = df['hip_angle'].values if 'hip_angle' in df.columns else np.full_like(timestamps, np.nan)
    elbow = df['elbow_angle'].values if 'elbow_angle' in df.columns else np.full_like(timestamps, np.nan)
    
    # We use percentiles (98th max, 2nd min) to ignore sensor noise spikes.
    gt_metrics = {}
    
    # Knee Metrics
    valid_knee = knee[~np.isnan(knee)]
    if len(valid_knee) > 10:
        gt_metrics['max_knee_ext'] = np.percentile(valid_knee, 98)
        gt_metrics['min_knee_flex'] = np.percentile(valid_knee, 2)
    else:
        return None, None, None # Skip empty files
        
    # Hip Metrics
    valid_hip = hip[~np.isnan(hip)]
    if len(valid_hip) > 10:
        gt_metrics['min_hip_angle'] = np.percentile(valid_hip, 2)
        gt_metrics['avg_hip_angle'] = np.mean(valid_hip)
    else:
        gt_metrics['min_hip_angle'] = np.nan
        gt_metrics['avg_hip_angle'] = np.nan

    # Elbow Metrics
    valid_elbow = elbow[~np.isnan(elbow)]
    if len(valid_elbow) > 10:
        gt_metrics['avg_elbow_angle'] = np.mean(valid_elbow)
    else:
        gt_metrics['avg_elbow_angle'] = np.nan
        
    joint_data = {
        'knee': knee,
        'hip': hip,
        'elbow': elbow
    }
    
    return timestamps, joint_data, gt_metrics

class ALSimExperiment:
    def __init__(self, timestamps, y_values, kernel_type='matern', acq_strategy='uncertainty'):
        self.timestamps = timestamps
        self.y_values = y_values
        self.total_frames = len(timestamps)
        
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

    def initialize_with_indices(self, indices):
        self.visited_indices.extend(indices)
        valid_idx = [i for i in indices if not np.isnan(self.y_values[i])]
        self.train_indices.extend(valid_idx)
        wasted = [i for i in indices if np.isnan(self.y_values[i])]
        self.wasted_indices.extend(wasted)

        # Optimize hyperparameters on the first batch
        self.update_model(fit=True)

    def update_model(self, fit=False):
        if len(self.train_indices) == 0:
            return

        X_train = self.timestamps[self.train_indices].reshape(-1, 1)
        Y_train = self.y_values[self.train_indices].reshape(-1, 1)
        
        X_train_norm = torch.tensor(self.x_scaler.transform(X_train), dtype=torch.double)
        Y_train_norm = torch.tensor(self.y_scaler.fit_transform(Y_train), dtype=torch.double)
        
        time_std = self.x_scaler.scale_[0] 

        # preserve old state dict if not fitting
        old_state_dict = None
        if self.model is not None and not fit:
            # We save the learned parameters (lenghtscale, noise, etc)
            # We filter out 'train inputs' because the data size is changing
            old_state_dict = self.model.state_dict()
            old_state_dict = {k : v for k, v in old_state_dict.items() if 'train_inputs' not in k and 'train_targets' not in k}
        
        # Priors
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

        if self.kernel_type == 'rbf':
            covar = ScaleKernel(RBFKernel(lengthscale_prior=fast_ls_prior))
            
        elif self.kernel_type == 'locally_periodic':
            per_kernel = PeriodicKernel(period_length_prior=per_prior, lengthscale_prior=smooth_prior)
            per_kernel.period_length = fit_period_mean
            per_kernel.lengthscale = smooth_target
            mat_kernel = MaternKernel(nu=2.5, lengthscale_prior=slow_ls_prior)
            covar = ScaleKernel(per_kernel * mat_kernel)
            
        else:
            covar = ScaleKernel(MaternKernel(nu=2.5, lengthscale_prior=fast_ls_prior))
            
        self.model = SingleTaskGP(X_train_norm, Y_train_norm, covar_module=covar)

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
            
            if stacked.shape[0] == 2:
                weights = torch.tensor([0.8, 0.2], dtype=torch.double)
                weighted_variances = stacked * weights.view(-1, 1)
                final_variance = torch.sum(weighted_variances, dim=0)
            else:
                final_variance, _ = torch.max(stacked, dim=0)
        else:
            final_variance = my_variance

        # Wasted index exclusion logic
        if len(self.wasted_indices) > 0:
            candidate_times = torch.tensor(self.timestamps[candidate_indices], dtype=torch.float32)
            wasted_times = torch.tensor(self.timestamps[self.wasted_indices], dtype=torch.float32)
            dists = torch.abs(candidate_times.unsqueeze(1) - wasted_times.unsqueeze(0))
            min_dists, _ = torch.min(dists, dim=1)
            suppression_radius = 1.0 
            too_close_mask = min_dists < suppression_radius
            final_variance[too_close_mask] = -1.0
            if torch.max(final_variance) == -1.0:
                 final_variance[too_close_mask] = 0.0

        best_local_idx = torch.argmax(final_variance).item()
        return candidate_indices[best_local_idx]

    def add_observation(self, idx, fit=False):
        if idx in self.visited_indices: return False
        self.visited_indices.append(idx)
        val = self.y_values[idx]
        if np.isnan(val):
            self.wasted_indices.append(idx)
            return False
        else:
            self.train_indices.append(idx)
            self.update_model(fit=fit) # because we optimize the hyperparameters every Nth sample
            return True

    def predict_curve(self):
        if self.model is None: return None
        self.model.eval()
        with torch.no_grad():
            posterior = self.model(self.X_all_norm)
            pred_y_norm = posterior.mean
        return self.y_scaler.inverse_transform(pred_y_norm.numpy().reshape(-1, 1)).flatten()

# ==========================================
# 3. MULTI-FILE BENCHMARK LOOP
# ==========================================

configs = [
    # {'name': 'Joint Uncertainty (Matern)', 'kernel': 'matern', 'acq': 'joint_uncertainty'},
    # {'name': 'Single Uncertainty (Matern)', 'kernel': 'matern', 'acq': 'uncertainty'},
    # {'name': 'Random (Matern)', 'kernel': 'matern', 'acq': 'random'},
    # {'name': 'Joint Uncertainty (LocPer)', 'kernel': 'locally_periodic', 'acq': 'joint_uncertainty'},
    # {'name': 'Single Uncertainty (LocPer)', 'kernel': 'locally_periodic', 'acq': 'uncertainty'},
    {'name': 'Joint Uncertainty (RBF)', 'kernel': 'rbf', 'acq': 'joint_uncertainty'},
    # {'name': 'Single Uncertainty (RBF)', 'kernel': 'rbf', 'acq': 'uncertainty'},

    # {
    #     'name': 'Hybrid (RBF-Knee / Matern-Hip)', 
    #     'kernel_knee': 'rbf',      # High precision for Knee
    #     'kernel_hip': 'matern',    # Robustness for Hip
    #     'kernel_elbow': 'matern',  # Robustness for Elbow
    #     'acq': 'joint_uncertainty'
    # },
]

# --- CONFIGURATION ---
CSV_FOLDER = "processed_results"  # Folder where your CSVs are stored
N_SAMPLES = 30 
INIT_SAMPLES = 5
FIT_INTERVAL = 5  # optimize every 5th sample
N_RUNS = 1 # Number of times to repeat the experiment per video with different seed for initial points

# Find all CSVs
csv_files = glob.glob(os.path.join(CSV_FOLDER, "*_data.csv"))

if not csv_files:
    print(f"No CSV files found in '{CSV_FOLDER}'. Please check the path.")
    exit()

all_results = []
print(f"Starting Benchmark across {len(csv_files)} video files...")

for i, csv_path in enumerate(csv_files):
    filename = os.path.basename(csv_path)
    
    # 1. Load Data & Calculate Dynamic Ground Truth
    timestamps, joint_data, gt_metrics = load_real_data_dynamic(csv_path)
    
    if timestamps is None:
        print(f"Skipping {filename} (Invalid or Empty)")
        continue
        
    print(f"\n[Video {i+1}/{len(csv_files)}] {filename}")
    print(f"  GT Knee Max: {gt_metrics['max_knee_ext']:.1f}° | GT Hip Min: {gt_metrics['min_hip_angle']:.1f}°")

    # Loop over runs
    for run_id in range(N_RUNS):

        # Setup Initial Indices
        # Create unique, reproducible seed for this video + Run combo
        seed = 42 + i + (run_id *  100)

        np.random.seed(seed)
        valid_pool = np.where(~np.isnan(joint_data['knee']))[0]
        
        if len(valid_pool) < INIT_SAMPLES:
            print("  Not enough valid data points.")
            continue
            
        fixed_init_idx = np.random.choice(valid_pool, INIT_SAMPLES, replace=False).tolist()

        # 3. Test Each Config
        for config in configs:

            # Determine kernel for each joint (hybrid vs standard approach)
            # If 'kernel' is in config, use it as default for all. Otherwise, look for specific joint-specific kernels.

            default_kernel = config.get('kernel', 'matern')
            k_knee = config.get('kernel_knee', default_kernel)
            k_hip = config.get('kernel_hip', default_kernel)
            k_elbow = config.get('kernel_elbow', default_kernel)

            exp_knee = ALSimExperiment(timestamps, joint_data['knee'], k_knee, config['acq'])
            exp_hip = ALSimExperiment(timestamps, joint_data['hip'], k_hip, config['acq'])
            exp_elbow = ALSimExperiment(timestamps, joint_data['elbow'], k_elbow, config['acq'])
            
            exp_knee.initialize_with_indices(fixed_init_idx)
            exp_hip.initialize_with_indices(fixed_init_idx)
            exp_elbow.initialize_with_indices(fixed_init_idx)
            
            valid_cnt = 0

            pbar_desc = f"  Run {run_id+1}/{N_RUNS} | {config['name']}"    

            with tqdm(total=N_SAMPLES, desc=pbar_desc, leave=False, unit="sample") as pbar:
                while valid_cnt < N_SAMPLES:
                    next_idx = exp_knee.select_next_point(other_experiments=[exp_hip])
                    if next_idx is None: break

                    do_fit = ((valid_cnt + 1) % FIT_INTERVAL == 0) # we optimize hyperparams every Nth sample
                        
                    if exp_knee.add_observation(next_idx, fit=do_fit):
                        valid_cnt += 1
                        pbar.update(1)
                    exp_hip.add_observation(next_idx, fit=do_fit)
                    exp_elbow.add_observation(next_idx, fit=do_fit)
                    
                    # Record Metrics
                    row = {
                        'Video': filename,
                        'Run_ID': run_id,
                        'Config': config['name'],
                        'Query_Step': valid_cnt
                    }
                    
                    pred_knee = exp_knee.predict_curve()
                    if pred_knee is not None:
                        row['Err_Max_Knee_Ext'] = abs(np.max(pred_knee) - gt_metrics['max_knee_ext'])
                        row['Err_Min_Knee_Flex'] = abs(np.min(pred_knee) - gt_metrics['min_knee_flex'])
                    print("MAX KNEE PRED:", np.max(pred_knee))
                        
                    pred_hip = exp_hip.predict_curve()
                    if pred_hip is not None:
                        row['Err_Min_Hip_Angle'] = abs(np.min(pred_hip) - gt_metrics['min_hip_angle'])
                        row['Err_Avg_Hip_Angle'] = abs(np.mean(pred_hip) - gt_metrics['avg_hip_angle'])
                    
                    pred_elbow = exp_elbow.predict_curve()
                    if pred_elbow is not None:
                        row['Err_Avg_Elbow_Angle'] = abs(np.mean(pred_elbow) - gt_metrics['avg_elbow_angle'])

                    all_results.append(row)

# ==========================================
# 4. AGGREGATED VISUALIZATION
# ==========================================
if not all_results:
    print("No results collected.")
    exit()

df_res = pd.DataFrame(all_results)

# Print Summary Table (Last Step Performance)
final_step = df_res[df_res['Query_Step'] == N_SAMPLES]
# summary = final_step.groupby('Config')[['Err_Max_Knee_Ext', 'Err_Min_Hip_Angle']].agg(['mean', 'std'])
summary = final_step.groupby('Config')[['Err_Max_Knee_Ext', 'Err_Min_Hip_Angle', 'Err_Min_Knee_Flex', 'Err_Avg_Elbow_Angle', 'Err_Avg_Hip_Angle']].agg(['mean', 'std'])

print("\n=== FINAL BENCHMARK SUMMARY (Avg across all videos) ===")
print(summary)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=df_res, x='Query_Step', y='Err_Max_Knee_Ext', hue='Config', marker='o')
plt.title('Max Knee Extension Error (Avg over Videos)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.lineplot(data=df_res, x='Query_Step', y='Err_Min_Hip_Angle', hue='Config', marker='o')
plt.title('Min Hip Angle Error (Avg over Videos)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize Final Fit for the LAST video processed (as a sanity check)
def plot_joint_fit(ax, exp, joint_name, true_time, true_vals, gt_val=None):
    if exp.model is None: return
    
    exp.model.eval()
    with torch.no_grad():
        posterior = exp.model(exp.X_all_norm)
        mean = posterior.mean
        lower, upper = posterior.confidence_region()

    pred_mean = exp.y_scaler.inverse_transform(mean.numpy().reshape(-1, 1)).flatten()
    pred_lower = exp.y_scaler.inverse_transform(lower.numpy().reshape(-1, 1)).flatten()
    pred_upper = exp.y_scaler.inverse_transform(upper.numpy().reshape(-1, 1)).flatten()

    mask = ~np.isnan(true_vals)
    ax.scatter(true_time[mask], true_vals[mask], c='gray', s=10, alpha=0.4, label='Data')
    ax.plot(true_time, pred_mean, 'b-', linewidth=2, label='GP Mean')
    ax.fill_between(true_time, pred_lower, pred_upper, color='blue', alpha=0.1, label='Confidence')

    sampled_t = true_time[exp.train_indices]
    sampled_y = true_vals[exp.train_indices]
    ax.scatter(sampled_t, sampled_y, c='red', s=40, marker='x', zorder=5, label='Sampled')
    
    if gt_val:
        ax.axhline(gt_val, color='green', linestyle='--', label=f'True Max ({gt_val:.1f})')

    ax.set_title(f"Final {joint_name} Fit")
    ax.set_ylabel("Angle (deg)")
    ax.legend(loc='upper right', fontsize='small')

plt.figure(figsize=(15, 5))
ax1 = plt.subplot(1, 3, 1)
plot_joint_fit(ax1, exp_knee, "Knee", timestamps, joint_data['knee'], gt_metrics['max_knee_ext'])
ax2 = plt.subplot(1, 3, 2)
plot_joint_fit(ax2, exp_hip, "Hip", timestamps, joint_data['hip'], gt_metrics['min_hip_angle'])
ax3 = plt.subplot(1, 3, 3)
plot_joint_fit(ax3, exp_elbow, "Elbow", timestamps, joint_data['elbow'])
plt.tight_layout()
plt.show()