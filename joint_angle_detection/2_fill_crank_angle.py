import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C, 
    ExpSineSquared, Matern
)
from sklearn.preprocessing import StandardScaler
'''
This script tries to clean the data by removing low-confidence or high-confidence erroneous points
Then uses a GP to attempt at filling in missing crank angle
'''

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="gp", choices=["linear", "gp"])
    return p.parse_args()

def remove_erroneous_points(df: pd.DataFrame, min_conf: float = 0.8) -> pd.DataFrame:
    df = df.copy()
    
    required_cols = [
        'detected_knee_angle', 'detected_hip_angle', 'detected_elbow_angle',
        'foot_x', 'foot_y', 'opposite_foot_x', 'opposite_foot_y',
        'foot_conf', 'opposite_foot_conf'
    ]

    #if bike_angle_deg is in df, include
    if 'bike_angle_deg' in df.columns:
        required_cols.insert(0, 'bike_angle_deg')
    
    # Remove rows with missing required features
    missing_features = df[required_cols].isna().any(axis=1)
    df = df[~missing_features].reset_index(drop=True)
    
    # Remove rows with low confidence
    low_conf = (df["foot_conf"] < min_conf) | (df["opposite_foot_conf"] < min_conf)
    df.loc[low_conf, "detected_crank_angle"] = np.nan
    
    print(f"Cleaned: {missing_features.sum()} dropped, {low_conf.sum()} set to NaN")
    
    return df

def interpolate_crank_angle_linear(video_df: pd.DataFrame) -> pd.DataFrame:
    result_df = video_df.copy()
    
    angles = result_df["detected_crank_angle"].values
    valid_mask = ~pd.isna(angles)
    missing_mask = ~valid_mask
    
    result_df["crank_angle_variance"] = np.nan
    
    angles_float = np.where(valid_mask, angles, np.nan)
    angles_rad = np.deg2rad(angles_float)
    sin_vals = np.sin(angles_rad)
    cos_vals = np.cos(angles_rad)
    
    frame_nums = result_df["frame_number"].values
    sin_series = pd.Series(sin_vals, index=frame_nums)
    cos_series = pd.Series(cos_vals, index=frame_nums)
    
    sin_pred = sin_series.interpolate(method="linear").values
    cos_pred = cos_series.interpolate(method="linear").values
    
    filled_angles = np.rad2deg(np.arctan2(sin_pred, cos_pred))
    filled_angles = filled_angles % 360
    
    result_df["crank_angle_filled"] = filled_angles
    result_df["crank_angle_interpolated"] = missing_mask
    
    print(f"    Linear: filled {missing_mask.sum()} missing values")
    
    return result_df


def interpolate_crank_angle_gp(video_df: pd.DataFrame) -> pd.DataFrame:
    result_df = video_df.copy()
    result_df["crank_angle_variance"] = 0.0
    
    feature_cols = [
        'detected_knee_angle', 'detected_hip_angle', 'detected_elbow_angle',
        'foot_x', 'foot_y',
        'opposite_foot_x', 'opposite_foot_y'
    ]
    
    #if we have real or interpolated bike angle, include
    if 'bike_angle_deg' in result_df.columns:
        feature_cols.insert(0, 'bike_angle_deg')

    valid_mask = ~pd.isna(result_df["detected_crank_angle"])
    missing_mask = ~valid_mask
    
    #edge case of no missing
    if missing_mask.sum() == 0:
        result_df["crank_angle_filled"] = result_df["detected_crank_angle"]
        result_df["crank_angle_interpolated"] = False
        return result_df
    
    X_train = result_df.loc[valid_mask, feature_cols].values
    y_angle_train = result_df.loc[valid_mask, "detected_crank_angle"].values
    
    y_rad_train = np.radians(y_angle_train)
    sin_train = np.sin(y_rad_train)
    cos_train = np.cos(y_rad_train)
    
    X_pred = result_df.loc[missing_mask, feature_cols].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)
    
    #params:
    periodici = 25
    noise = .05

    kernel = (
        C(.5) * ExpSineSquared(length_scale=1.0, periodicity=periodici) +
        WhiteKernel(noise_level=noise)
    )
    
    gp_sin = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=6)
    gp_cos = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=6)
    
    gp_sin.fit(X_train_scaled, sin_train)
    gp_cos.fit(X_train_scaled, cos_train)
    
    sin_pred, sin_std = gp_sin.predict(X_pred_scaled, return_std=True)
    cos_pred, cos_std = gp_cos.predict(X_pred_scaled, return_std=True)
    
    angle_pred_rad = np.arctan2(sin_pred, cos_pred)
    angle_pred_deg = (np.degrees(angle_pred_rad) + 360) % 360
    
    angle_variance = (sin_std**2 + cos_std**2) / 2
    
    result_df["crank_angle_filled"] = result_df["detected_crank_angle"].copy()
    result_df["crank_angle_interpolated"] = False
    result_df.loc[missing_mask, "crank_angle_filled"] = angle_pred_deg
    result_df.loc[missing_mask, "crank_angle_interpolated"] = True
    result_df.loc[missing_mask, "crank_angle_variance"] = angle_variance
    
    print(f"    GP filled {missing_mask.sum()}, var={np.mean(angle_variance):.4f}")
    
    return result_df


def interpolate_crank_angle(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    df = df.copy()
    df["crank_angle_filled"] = df["detected_crank_angle"].copy()
    df["crank_angle_interpolated"] = False
    
    videos = df["source_video"].unique()
    
    results = []
    for video_name in videos:
        print(f"  {video_name}:")
        video_df = df[df["source_video"] == video_name].copy()
        
        if method == "linear":
            video_df = interpolate_crank_angle_linear(video_df)
        elif method == "gp":
            video_df = interpolate_crank_angle_gp(video_df)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        results.append(video_df)
    
    return pd.concat(results, ignore_index=True)


def main():
    args = parse_args()
    
    input_path = Path("output/synchronized_dataset.csv")
    output_path = Path("output/synchronized_dataset_filled.csv")
    
    df = pd.read_csv(input_path)
    df = remove_erroneous_points(df, min_conf=0.8)
    
    if "detected_crank_angle" not in df.columns:
        print("Error: 'detected_crank_angle' not found")
        return

    n_missing = df["detected_crank_angle"].isna().sum()
    n_total = len(df)
    print(f"{len(df)} rows, {n_missing} missing ({100*n_missing/n_total:.1f}%)")
    
    if n_missing == 0:
        df.to_csv(output_path, index=False)
        return
    
    df = interpolate_crank_angle(df, method=args.method)
    
    crank_rad = np.deg2rad(df["crank_angle_filled"])
    df["crank_sin"] = np.sin(crank_rad)
    df["crank_cos"] = np.cos(crank_rad)
    
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
