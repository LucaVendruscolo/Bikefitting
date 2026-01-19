import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    #defaults to first video in dataset
    p.add_argument("--video", type=str, required=False)
    return p.parse_args()


def plot_angles(df: pd.DataFrame, video_name: str):
    crank_col = "crank_angle_filled" if "crank_angle_filled" in df.columns else "detected_crank_angle"
    
    mask = ~pd.isna(df[crank_col]) & ~pd.isna(df["detected_knee_angle"])
    crank = df.loc[mask, crank_col].values
    knee = df.loc[mask, "detected_knee_angle"].values
    frames = df.loc[mask, "frame_number"].values
    
    if len(crank) == 0:
        return
    
    crank_ma = pd.Series(crank).rolling(window=5, center=True, min_periods=1).mean().values
    knee_ma = pd.Series(knee).rolling(window=5, center=True, min_periods=1).mean().values
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(frames, crank, 'b-', alpha=0.2, linewidth=0.8, label='Crank (raw)')
    ax1.plot(frames, crank_ma, 'b-', linewidth=2, label='Crank (MA)')
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("Crank Angle (°)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 360)
    
    ax2.plot(frames, knee, 'r-', alpha=0.2, linewidth=0.8, label='Knee (raw)')
    ax2.plot(frames, knee_ma, 'r-', linewidth=2, label='Knee (MA)')
    ax2.set_ylabel("Knee Angle (°)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title(f"Crank & Knee Angles - {video_name}")
    ax1.grid(True, alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    output_path = Path("output") / f"angles_{video_name.replace('.MOV', '')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_crank_with_variance(df: pd.DataFrame, video_name: str):
    crank_col = "crank_angle_filled" if "crank_angle_filled" in df.columns else "detected_crank_angle"
    
    mask = ~pd.isna(df[crank_col])
    crank = df.loc[mask, crank_col].values
    frames = df.loc[mask, "frame_number"].values
    variance = df.loc[mask, "crank_angle_variance"].values if "crank_angle_variance" in df.columns else None
    
    if len(crank) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(frames, crank, 'b-', alpha=0.3, linewidth=0.8, label='Crank (raw)')
    
    crank_ma = pd.Series(crank).rolling(window=5, center=True, min_periods=1).mean().values
    ax.plot(frames, crank_ma, 'b-', linewidth=2, label='Crank (MA)')
    
    if variance is not None and not np.all(np.isnan(variance)):
        std = np.sqrt(np.nan_to_num(variance, nan=0.0))
        std_deg = std * 60
        ax.fill_between(frames, crank_ma - std_deg, crank_ma + std_deg, 
                        alpha=0.3, color='blue', label='±1 std (GP)')
    
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Crank Angle (°)")
    ax.set_ylim(0, 360)
    ax.set_title(f"Crank Angle with Uncertainty - {video_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = Path("output") / f"crank_variance_{video_name.replace('.MOV', '')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    df = pd.read_csv("output/synchronized_dataset_filled.csv")
    if args.video:
        video_name = args.video
    else:
        #take first if not included as arg
        video_name = df["source_video"].iloc[0]

    df = df[df["source_video"] == video_name]

    plot_angles(df,video_name)
    plot_crank_with_variance(df,video_name)


if __name__ == "__main__":
    main()
