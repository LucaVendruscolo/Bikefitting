"""
BikeDataset Builder Tool
========================
A GUI application to synchronize video frames with IMU sensor data.

Workflow:
1. Select a video from the Training_Videos folder
2. Play/pause to find the frame showing the phone time
3. Enter the time displayed on the phone screen
4. Select a matching CSV file from the Runs folder
5. Trim the video to exclude setup periods
6. Repeat for all videos
7. Click "Create Dataset" to export synchronized frames and CSV

Usage:
    python dataset_builder.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import shutil
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import threading


@dataclass
class VideoSyncConfig:
    """Configuration for a single video synchronization."""
    video_path: str
    csv_path: str
    phone_time: str  # Time shown on phone screen (ISO format)
    sync_frame: int  # Frame number where phone time is visible
    trim_start_frame: int  # First frame to include
    trim_end_frame: int  # Last frame to include
    video_fps: float
    total_frames: int
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class VideoPlayer:
    """Handles video playback and frame extraction."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        
    def get_frame(self, frame_num: int) -> Optional[tuple]:
        """Get a specific frame. Returns (frame, frame_num) or None."""
        if frame_num < 0 or frame_num >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_num
            return frame, frame_num
        return None
    
    def get_next_frame(self) -> Optional[tuple]:
        """Get next frame. Returns (frame, frame_num) or None."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            return frame, self.current_frame
        return None
    
    def frame_to_time_offset(self, frame_num: int) -> float:
        """Convert frame number to time offset in seconds."""
        return frame_num / self.fps
    
    def close(self):
        self.cap.release()


class DatasetBuilderApp:
    """Main application for building synchronized datasets."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BikeDataset Builder")
        self.root.geometry("1400x900")
        
        # Paths
        self.base_path = Path(__file__).parent
        self.videos_path = self.base_path / "Training_Videos"
        self.runs_path = self.base_path / "Runs"
        self.config_path = self.base_path / "sync_configs.json"
        
        # State
        self.video_player: Optional[VideoPlayer] = None
        self.current_photo = None
        self.is_playing = False
        self.play_thread = None
        self.sync_configs: Dict[str, VideoSyncConfig] = {}
        self._updating_slider = False  # Prevent recursive slider updates
        
        # Load existing configs
        self.load_configs()
        
        # Build UI
        self.setup_ui()
        
        # Populate lists
        self.refresh_video_list()
        self.refresh_csv_list()
        
    def setup_ui(self):
        """Build the user interface."""
        # Configure grid weights
        self.root.columnconfigure(0, weight=0)  # Left panel
        self.root.columnconfigure(1, weight=1)  # Video panel
        self.root.rowconfigure(0, weight=1)
        
        # === Left Panel (File Selection) ===
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew")
        
        # Video List
        ttk.Label(left_frame, text="Videos:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        video_list_frame = ttk.Frame(left_frame)
        video_list_frame.pack(fill="both", expand=True, pady=(5, 10))
        
        self.video_listbox = tk.Listbox(video_list_frame, width=35, height=12, 
                                         selectmode=tk.SINGLE, font=("Consolas", 10))
        video_scrollbar = ttk.Scrollbar(video_list_frame, orient="vertical", 
                                         command=self.video_listbox.yview)
        self.video_listbox.configure(yscrollcommand=video_scrollbar.set)
        self.video_listbox.pack(side="left", fill="both", expand=True)
        video_scrollbar.pack(side="right", fill="y")
        self.video_listbox.bind("<<ListboxSelect>>", self.on_video_select)
        
        # CSV List (for reference - auto-detection is primary)
        ttk.Label(left_frame, text="IMU CSV Files (auto-detected):", font=("Arial", 11, "bold")).pack(anchor="w")
        
        csv_list_frame = ttk.Frame(left_frame)
        csv_list_frame.pack(fill="both", expand=True, pady=(5, 10))
        
        self.csv_listbox = tk.Listbox(csv_list_frame, width=35, height=10, 
                                       selectmode=tk.SINGLE, font=("Consolas", 10))
        csv_scrollbar = ttk.Scrollbar(csv_list_frame, orient="vertical", 
                                       command=self.csv_listbox.yview)
        self.csv_listbox.configure(yscrollcommand=csv_scrollbar.set)
        self.csv_listbox.pack(side="left", fill="both", expand=True)
        csv_scrollbar.pack(side="right", fill="y")
        
        # Auto-detected CSV label
        self.auto_csv_label = ttk.Label(left_frame, text="Auto-detected: None", 
                                         font=("Arial", 10), foreground="gray")
        self.auto_csv_label.pack(anchor="w", pady=(0, 5))
        
        # Sync Status
        ttk.Label(left_frame, text="Sync Status:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 5))
        
        self.status_text = tk.Text(left_frame, width=35, height=8, font=("Consolas", 9),
                                   state="disabled", bg="#f0f0f0")
        self.status_text.pack(fill="x")
        
        # Action Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill="x", pady=(15, 5))
        
        ttk.Button(btn_frame, text="Save Progress", 
                   command=self.save_configs).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Create Dataset", 
                   command=self.create_dataset).pack(fill="x", pady=2)
        
        # === Right Panel (Video Player) ===
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Video Display
        video_frame = ttk.LabelFrame(right_frame, text="Video Preview", padding=5)
        video_frame.grid(row=0, column=0, sticky="nsew")
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="Select a video to begin",
                                      anchor="center", background="#2a2a2a")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Video Info
        self.video_info_label = ttk.Label(video_frame, text="", font=("Arial", 10))
        self.video_info_label.grid(row=1, column=0, sticky="w", pady=(5, 0))
        
        # Controls Frame
        controls_frame = ttk.LabelFrame(right_frame, text="Controls", padding=10)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        
        # Playback controls
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(fill="x", pady=5)
        
        self.play_btn = ttk.Button(playback_frame, text="▶ Play", command=self.toggle_play, width=10)
        self.play_btn.pack(side="left", padx=2)
        
        ttk.Button(playback_frame, text="⏮ -10s", command=lambda: self.seek_relative(-10), width=8).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="◀ -1s", command=lambda: self.seek_relative(-1), width=8).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="◀ -1f", command=lambda: self.seek_frames(-1), width=8).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="▶ +1f", command=lambda: self.seek_frames(1), width=8).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="+1s ▶", command=lambda: self.seek_relative(1), width=8).pack(side="left", padx=2)
        ttk.Button(playback_frame, text="+10s ⏭", command=lambda: self.seek_relative(10), width=8).pack(side="left", padx=2)
        
        # Frame slider
        slider_frame = ttk.Frame(controls_frame)
        slider_frame.pack(fill="x", pady=5)
        
        ttk.Label(slider_frame, text="Frame:").pack(side="left")
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal",
                                       command=self.on_slider_change)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=5)
        
        self.frame_label = ttk.Label(slider_frame, text="0 / 0", width=15)
        self.frame_label.pack(side="left")
        
        # Sync Settings Frame
        sync_frame = ttk.LabelFrame(right_frame, text="Synchronization Settings", padding=10)
        sync_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        # Phone time input
        time_frame = ttk.Frame(sync_frame)
        time_frame.pack(fill="x", pady=5)
        
        ttk.Label(time_frame, text="Phone Time (from video):").pack(side="left")
        self.phone_time_entry = ttk.Entry(time_frame, width=25, font=("Consolas", 11))
        self.phone_time_entry.pack(side="left", padx=10)
        ttk.Button(time_frame, text="🔍 Find CSV", command=self.find_matching_csv).pack(side="left", padx=5)
        ttk.Label(time_frame, text="Format: HH:MM:SS.mm (e.g., 15:23:45.67)", 
                  font=("Arial", 9), foreground="gray").pack(side="left")
        
        # Mark sync frame button
        sync_btn_frame = ttk.Frame(sync_frame)
        sync_btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(sync_btn_frame, text="📍 Mark Current Frame as Sync Point",
                   command=self.mark_sync_frame).pack(side="left")
        self.sync_frame_label = ttk.Label(sync_btn_frame, text="Sync frame: Not set", 
                                           font=("Arial", 10))
        self.sync_frame_label.pack(side="left", padx=20)
        
        # Trim Settings Frame
        trim_frame = ttk.LabelFrame(right_frame, text="Trim Video (Optional)", padding=10)
        trim_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        
        trim_btn_frame = ttk.Frame(trim_frame)
        trim_btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(trim_btn_frame, text="✂ Set Trim Start (Current Frame)",
                   command=self.set_trim_start).pack(side="left", padx=5)
        ttk.Button(trim_btn_frame, text="✂ Set Trim End (Current Frame)",
                   command=self.set_trim_end).pack(side="left", padx=5)
        ttk.Button(trim_btn_frame, text="↺ Reset Trim",
                   command=self.reset_trim).pack(side="left", padx=5)
        
        self.trim_label = ttk.Label(trim_frame, text="Trim: Full video (no trim set)", 
                                     font=("Arial", 10))
        self.trim_label.pack(anchor="w", pady=5)
        
        # Save current video config
        save_frame = ttk.Frame(right_frame)
        save_frame.grid(row=4, column=0, sticky="ew", pady=(15, 0))
        
        self.save_video_btn = ttk.Button(save_frame, text="💾 Save This Video's Configuration",
                                          command=self.save_current_video_config)
        self.save_video_btn.pack(side="left")
        
        self.config_status_label = ttk.Label(save_frame, text="", font=("Arial", 10))
        self.config_status_label.pack(side="left", padx=20)
        
        # Initialize state
        self.current_sync_frame = None
        self.trim_start = None
        self.trim_end = None
        self.current_video_name = None  # Track currently loaded video
        
    def refresh_video_list(self):
        """Refresh the list of videos."""
        self.video_listbox.delete(0, tk.END)
        if self.videos_path.exists():
            videos = sorted([f.name for f in self.videos_path.iterdir() 
                            if f.suffix.lower() in ['.mov', '.mp4', '.avi', '.mkv']])
            for video in videos:
                # Add checkmark if configured
                prefix = "✓ " if video in self.sync_configs else "  "
                self.video_listbox.insert(tk.END, prefix + video)
                
    def refresh_csv_list(self):
        """Refresh the list of CSV files."""
        self.csv_listbox.delete(0, tk.END)
        if self.runs_path.exists():
            csvs = sorted([f.name for f in self.runs_path.iterdir() 
                          if f.suffix.lower() == '.csv'])
            for csv in csvs:
                self.csv_listbox.insert(tk.END, csv)
                
    def on_video_select(self, event):
        """Handle video selection."""
        selection = self.video_listbox.curselection()
        if not selection:
            return
            
        # Get video name (remove prefix)
        video_name = self.video_listbox.get(selection[0]).strip()
        if video_name.startswith("✓ "):
            video_name = video_name[2:]
        
        # If same video is re-selected, don't reset anything
        if video_name == self.current_video_name:
            return
            
        video_path = self.videos_path / video_name
        
        # Stop current playback
        self.is_playing = False
        if self.video_player:
            self.video_player.close()
            
        # Load new video
        try:
            self.video_player = VideoPlayer(str(video_path))
            self.current_video_name = video_name  # Track current video
            self.frame_slider.configure(to=self.video_player.total_frames - 1)
            
            # Load existing config if available
            if video_name in self.sync_configs:
                config = self.sync_configs[video_name]
                self.current_sync_frame = config.sync_frame
                self.trim_start = config.trim_start_frame
                self.trim_end = config.trim_end_frame
                self.phone_time_entry.delete(0, tk.END)
                self.phone_time_entry.insert(0, config.phone_time)
                
                # Select the matching CSV
                csv_name = Path(config.csv_path).name
                for i in range(self.csv_listbox.size()):
                    if self.csv_listbox.get(i) == csv_name:
                        self.csv_listbox.selection_clear(0, tk.END)
                        self.csv_listbox.selection_set(i)
                        self.csv_listbox.see(i)
                        break
                
                self.auto_csv_label.config(text=f"✓ CSV: {csv_name}", foreground="green")
                self.update_trim_label()
                self.sync_frame_label.config(text=f"Sync frame: {self.current_sync_frame}")
                self.config_status_label.config(text="✓ Config loaded", foreground="green")
            else:
                # Reset state for new video
                self.current_sync_frame = None
                self.trim_start = 0
                self.trim_end = self.video_player.total_frames - 1
                self.phone_time_entry.delete(0, tk.END)
                self.csv_listbox.selection_clear(0, tk.END)
                self.auto_csv_label.config(text="Auto-detected: None", foreground="gray")
                self.update_trim_label()
                self.sync_frame_label.config(text="Sync frame: Not set")
                self.config_status_label.config(text="")
                
            # Show first frame
            self.show_frame(0)
            self.update_status()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            
    def show_frame(self, frame_num: int):
        """Display a specific frame."""
        if not self.video_player:
            return
            
        result = self.video_player.get_frame(frame_num)
        if result is None:
            return
            
        frame, actual_frame = result
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display (max 900x600)
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 900, 550
        scale = min(max_w / w, max_h / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            
        # Convert to PhotoImage
        img = Image.fromarray(frame_rgb)
        self.current_photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.current_photo)
        
        # Update labels (prevent recursive slider callback)
        self._updating_slider = True
        try:
            self.frame_slider.set(actual_frame)
        finally:
            self._updating_slider = False
        time_offset = self.video_player.frame_to_time_offset(actual_frame)
        self.frame_label.config(text=f"{actual_frame} / {self.video_player.total_frames - 1}")
        
        fps = self.video_player.fps
        self.video_info_label.config(
            text=f"Time: {time_offset:.2f}s | FPS: {fps:.1f} | "
                 f"Size: {self.video_player.width}x{self.video_player.height}"
        )
        
    def on_slider_change(self, value):
        """Handle slider movement."""
        # Skip if we're programmatically updating the slider (prevents recursion)
        if self._updating_slider:
            return
        if self.video_player and not self.is_playing:
            frame_num = int(float(value))
            self.show_frame(frame_num)
            
    def toggle_play(self):
        """Toggle play/pause."""
        if not self.video_player:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_btn.config(text="⏸ Pause")
            self.play_thread = threading.Thread(target=self.play_video, daemon=True)
            self.play_thread.start()
        else:
            self.play_btn.config(text="▶ Play")
            
    def play_video(self):
        """Play video in a separate thread."""
        import time
        
        while self.is_playing and self.video_player:
            current = self.video_player.current_frame
            if current >= self.video_player.total_frames - 1:
                self.is_playing = False
                self.root.after(0, lambda: self.play_btn.config(text="▶ Play"))
                break
                
            self.root.after(0, lambda f=current + 1: self.show_frame(f))
            time.sleep(1 / self.video_player.fps)
            
    def seek_relative(self, seconds: float):
        """Seek relative to current position."""
        if not self.video_player:
            return
        frames = int(seconds * self.video_player.fps)
        new_frame = max(0, min(self.video_player.current_frame + frames, 
                              self.video_player.total_frames - 1))
        self.show_frame(new_frame)
        
    def seek_frames(self, frames: int):
        """Seek by number of frames."""
        if not self.video_player:
            return
        new_frame = max(0, min(self.video_player.current_frame + frames, 
                              self.video_player.total_frames - 1))
        self.show_frame(new_frame)
        
    def mark_sync_frame(self):
        """Mark current frame as the sync point and auto-adjust trim if needed."""
        if not self.video_player:
            messagebox.showwarning("Warning", "Please select a video first.")
            return
            
        self.current_sync_frame = self.video_player.current_frame
        self.sync_frame_label.config(text=f"Sync frame: {self.current_sync_frame}")
        
        # Auto-adjust trim based on CSV time range
        self.auto_adjust_trim_to_csv()
        
    def auto_adjust_trim_to_csv(self):
        """Auto-adjust trim start/end based on CSV time range."""
        # Need: video player, sync frame, phone time, and CSV selected
        if not self.video_player or self.current_sync_frame is None:
            return
            
        phone_time_str = self.phone_time_entry.get().strip()
        if not phone_time_str:
            return
            
        # Get selected CSV
        csv_selection = self.csv_listbox.curselection()
        if not csv_selection:
            return
            
        csv_name = self.csv_listbox.get(csv_selection[0])
        csv_path = self.runs_path / csv_name
        
        try:
            # Parse phone time
            parsed_time = self.parse_phone_time(phone_time_str)
            
            # Load CSV time range
            csv_df = pd.read_csv(csv_path, usecols=['wall_time_iso'])
            csv_df['wall_time_iso'] = pd.to_datetime(csv_df['wall_time_iso'])
            csv_start_time = csv_df['wall_time_iso'].iloc[0]
            csv_end_time = csv_df['wall_time_iso'].iloc[-1]
            
            # Build full phone timestamp
            if parsed_time.year == 1900:
                phone_time_full = parsed_time.replace(
                    year=csv_start_time.year,
                    month=csv_start_time.month,
                    day=csv_start_time.day
                )
            else:
                phone_time_full = parsed_time
            
            phone_ts = pd.Timestamp(phone_time_full)
            fps = self.video_player.fps
            adjusted = False
            
            # Check if phone time is BEFORE CSV starts
            if phone_ts < csv_start_time:
                offset_seconds = (csv_start_time - phone_ts).total_seconds()
                csv_start_frame = self.current_sync_frame + int(offset_seconds * fps)
                
                if csv_start_frame > 0 and csv_start_frame < self.video_player.total_frames:
                    self.trim_start = csv_start_frame
                    adjusted = True
            
            # Check if video extends past CSV end
            if self.trim_end is None:
                self.trim_end = self.video_player.total_frames - 1
                
            trim_end_offset = (self.trim_end - self.current_sync_frame) / fps
            trim_end_time = phone_ts + timedelta(seconds=trim_end_offset)
            
            if trim_end_time > csv_end_time:
                offset_seconds = (csv_end_time - phone_ts).total_seconds()
                csv_end_frame = self.current_sync_frame + int(offset_seconds * fps)
                
                if csv_end_frame > 0 and csv_end_frame < self.video_player.total_frames:
                    self.trim_end = csv_end_frame
                    adjusted = True
            
            if adjusted:
                self.update_trim_label()
                
        except Exception as e:
            print(f"Warning: Could not auto-adjust trim: {e}")
    
    def set_trim_start(self):
        """Set trim start to current frame."""
        if not self.video_player:
            return
        self.trim_start = self.video_player.current_frame
        self.update_trim_label()
        
    def set_trim_end(self):
        """Set trim end to current frame."""
        if not self.video_player:
            return
        self.trim_end = self.video_player.current_frame
        self.update_trim_label()
        
    def reset_trim(self):
        """Reset trim to full video."""
        if not self.video_player:
            return
        self.trim_start = 0
        self.trim_end = self.video_player.total_frames - 1
        self.update_trim_label()
        
    def update_trim_label(self):
        """Update the trim label."""
        if self.video_player:
            start = self.trim_start if self.trim_start is not None else 0
            end = self.trim_end if self.trim_end is not None else self.video_player.total_frames - 1
            total = end - start + 1
            duration = total / self.video_player.fps
            self.trim_label.config(
                text=f"Trim: Frame {start} to {end} ({total} frames, {duration:.1f}s)"
            )
        else:
            self.trim_label.config(text="Trim: Full video (no trim set)")
            
    def save_current_video_config(self):
        """Save configuration for current video."""
        if not self.video_player:
            messagebox.showwarning("Warning", "Please select a video first.")
            return
            
        # Validate inputs
        phone_time = self.phone_time_entry.get().strip()
        if not phone_time:
            messagebox.showwarning("Warning", "Please enter the phone time.")
            return
            
        if self.current_sync_frame is None:
            messagebox.showwarning("Warning", "Please mark the sync frame.")
            return
            
        # Try to get selected CSV, or auto-detect
        csv_selection = self.csv_listbox.curselection()
        if csv_selection:
            csv_name = self.csv_listbox.get(csv_selection[0])
        else:
            # Try auto-detection
            csv_name = self.find_matching_csv()
            if not csv_name:
                return  # Error already shown by find_matching_csv
            
        csv_path = str(self.runs_path / csv_name)
        
        # Parse and validate phone time
        try:
            parsed_time = self.parse_phone_time(phone_time)
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid time format: {e}")
            return
            
        # Get video name from tracked state (not listbox selection which can be lost)
        if not self.current_video_name:
            messagebox.showwarning("Warning", "Please select a video first.")
            return
        video_name = self.current_video_name
        
        # Auto-adjust trim based on CSV time range
        trim_start = self.trim_start if self.trim_start is not None else 0
        trim_end = self.trim_end if self.trim_end is not None else self.video_player.total_frames - 1
        fps = self.video_player.fps
        
        try:
            # Load CSV to get time range
            csv_df = pd.read_csv(csv_path, usecols=['wall_time_iso'])
            csv_df['wall_time_iso'] = pd.to_datetime(csv_df['wall_time_iso'])
            csv_start_time = csv_df['wall_time_iso'].iloc[0]
            csv_end_time = csv_df['wall_time_iso'].iloc[-1]
            
            # Build full phone timestamp
            if parsed_time.year == 1900:
                phone_time_full = parsed_time.replace(
                    year=csv_start_time.year,
                    month=csv_start_time.month,
                    day=csv_start_time.day
                )
            else:
                phone_time_full = parsed_time
            
            phone_ts = pd.Timestamp(phone_time_full)
            auto_adjusted = False
            adjustment_msg = ""
            
            # Check if phone time is BEFORE CSV starts
            if phone_ts < csv_start_time:
                # Calculate offset in seconds
                offset_seconds = (csv_start_time - phone_ts).total_seconds()
                # Calculate frame where CSV starts
                csv_start_frame = self.current_sync_frame + int(offset_seconds * fps)
                
                if csv_start_frame > trim_start:
                    old_trim_start = trim_start
                    trim_start = csv_start_frame
                    auto_adjusted = True
                    adjustment_msg += f"\n\n⚠️ Auto-adjusted trim START:\n"
                    adjustment_msg += f"Phone time ({phone_time_full.strftime('%H:%M:%S.%f')[:-3]}) is before CSV starts ({csv_start_time.strftime('%H:%M:%S.%f')[:-3]})\n"
                    adjustment_msg += f"Trim start: frame {old_trim_start} → {trim_start}"
            
            # Check if phone time at trim_end is AFTER CSV ends
            trim_end_offset = (trim_end - self.current_sync_frame) / fps
            trim_end_time = phone_ts + timedelta(seconds=trim_end_offset)
            
            if trim_end_time > csv_end_time:
                # Calculate frame where CSV ends
                offset_seconds = (csv_end_time - phone_ts).total_seconds()
                csv_end_frame = self.current_sync_frame + int(offset_seconds * fps)
                
                if csv_end_frame < trim_end:
                    old_trim_end = trim_end
                    trim_end = csv_end_frame
                    auto_adjusted = True
                    adjustment_msg += f"\n\n⚠️ Auto-adjusted trim END:\n"
                    adjustment_msg += f"Video extends past CSV end ({csv_end_time.strftime('%H:%M:%S.%f')[:-3]})\n"
                    adjustment_msg += f"Trim end: frame {old_trim_end} → {trim_end}"
            
            if auto_adjusted:
                # Update the UI
                self.trim_start = trim_start
                self.trim_end = trim_end
                self.update_trim_label()
                
        except Exception as e:
            print(f"Warning: Could not auto-adjust trim: {e}")
            adjustment_msg = ""
            
        # Create config
        config = VideoSyncConfig(
            video_path=str(self.videos_path / video_name),
            csv_path=csv_path,
            phone_time=parsed_time.isoformat(),
            sync_frame=self.current_sync_frame,
            trim_start_frame=trim_start,
            trim_end_frame=trim_end,
            video_fps=fps,
            total_frames=self.video_player.total_frames
        )
        
        self.sync_configs[video_name] = config
        self.config_status_label.config(text="✓ Configuration saved!", foreground="green")
        self.refresh_video_list()
        self.update_status()
        self.save_configs()
        
        # Show confirmation with any auto-adjustments
        if adjustment_msg:
            messagebox.showinfo(
                "Configuration Saved",
                f"Video configuration saved successfully!{adjustment_msg}"
            )
        
    def parse_phone_time(self, time_str: str) -> datetime:
        """Parse phone time string to datetime.
        
        Handles 2-digit milliseconds from phone displays (e.g., 23:07:15.45)
        and converts them to 3-digit (23:07:15.450).
        """
        time_str = time_str.strip()
        
        # Handle 2-digit milliseconds: 23:07:15.45 -> 23:07:15.450
        # Check for pattern like HH:MM:SS.XX (2-digit ms)
        import re
        ms_match = re.match(r'^(\d{1,2}:\d{2}:\d{2})\.(\d{2})$', time_str)
        if ms_match:
            time_part, ms_part = ms_match.groups()
            # Convert 2-digit to 3-digit (45 -> 450, not 045)
            time_str = f"{time_part}.{ms_part}0"
        
        # Try full ISO format first
        try:
            return datetime.fromisoformat(time_str)
        except ValueError:
            pass
            
        # Try time-only formats
        formats = [
            "%H:%M:%S.%f",
            "%H:%M:%S",
            "%H:%M",
        ]
        
        for fmt in formats:
            try:
                t = datetime.strptime(time_str, fmt)
                # Assume today's date if only time provided
                # We'll use the date from the CSV file name later
                return t
            except ValueError:
                continue
                
        raise ValueError(f"Could not parse time: {time_str}")
    
    def get_csv_time_ranges(self) -> Dict[str, tuple]:
        """Get the time range for each CSV file. Returns {filename: (start_time, end_time)}"""
        csv_ranges = {}
        
        if not self.runs_path.exists():
            return csv_ranges
            
        for csv_file in self.runs_path.iterdir():
            if csv_file.suffix.lower() != '.csv':
                continue
                
            try:
                # Read just the first and last rows for efficiency
                df = pd.read_csv(csv_file, usecols=['wall_time_iso'])
                if len(df) == 0:
                    continue
                    
                start_time = pd.to_datetime(df['wall_time_iso'].iloc[0])
                end_time = pd.to_datetime(df['wall_time_iso'].iloc[-1])
                csv_ranges[csv_file.name] = (start_time, end_time)
            except Exception as e:
                print(f"Warning: Could not read {csv_file.name}: {e}")
                continue
                
        return csv_ranges
    
    def find_matching_csv(self):
        """Find the CSV file that matches the entered phone time.
        
        Automatically tries both AM and PM interpretations if time is ambiguous.
        """
        phone_time_str = self.phone_time_entry.get().strip()
        
        if not phone_time_str:
            messagebox.showwarning("Warning", "Please enter the phone time first.")
            return None
            
        try:
            phone_time = self.parse_phone_time(phone_time_str)
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid time format: {e}")
            return None
        
        # Get CSV time ranges
        csv_ranges = self.get_csv_time_ranges()
        
        if not csv_ranges:
            messagebox.showwarning("Warning", "No CSV files found in Runs folder.")
            return None
        
        # Create list of times to try: original and +12 hours (PM version)
        times_to_try = [phone_time]
        if phone_time.hour < 12:
            # Also try PM version (add 12 hours)
            pm_time = phone_time.replace(hour=phone_time.hour + 12)
            times_to_try.append(pm_time)
        elif phone_time.hour >= 12 and phone_time.hour < 24:
            # Also try AM version (subtract 12 hours)
            am_time = phone_time.replace(hour=phone_time.hour - 12)
            times_to_try.append(am_time)
        
        best_match = None
        best_score = float('inf')
        best_phone_time = phone_time
        
        for try_time in times_to_try:
            for csv_name, (start_time, end_time) in csv_ranges.items():
                # If phone_time only has time (no date), use the date from the CSV
                if try_time.year == 1900:  # Default year from time-only parsing
                    phone_time_full = try_time.replace(
                        year=start_time.year,
                        month=start_time.month,
                        day=start_time.day
                    )
                else:
                    phone_time_full = try_time
                
                # Convert to pandas Timestamp for comparison
                phone_ts = pd.Timestamp(phone_time_full)
                
                # Check if phone time is within the CSV's time range
                if start_time <= phone_ts <= end_time:
                    # Perfect match - time is within range
                    best_match = csv_name
                    best_score = 0
                    best_phone_time = try_time
                    break
                else:
                    # Calculate distance to the range
                    if phone_ts < start_time:
                        distance = (start_time - phone_ts).total_seconds()
                    else:
                        distance = (phone_ts - end_time).total_seconds()
                    
                    if distance < best_score:
                        best_score = distance
                        best_match = csv_name
                        best_phone_time = try_time
            
            if best_score == 0:
                break  # Found perfect match, stop searching
        
        if best_match:
            # Update UI to show the match
            self.auto_csv_label.config(
                text=f"✓ Auto-detected: {best_match}", 
                foreground="green"
            )
            
            # Select it in the listbox
            for i in range(self.csv_listbox.size()):
                if self.csv_listbox.get(i) == best_match:
                    self.csv_listbox.selection_clear(0, tk.END)
                    self.csv_listbox.selection_set(i)
                    self.csv_listbox.see(i)
                    break
            
            # If we used a different time (AM/PM conversion), update the entry
            if best_phone_time != phone_time:
                # Update the phone time entry with the corrected time
                corrected_time_str = best_phone_time.strftime("%H:%M:%S.%f")[:-4]  # Keep 2 decimal places
                self.phone_time_entry.delete(0, tk.END)
                self.phone_time_entry.insert(0, corrected_time_str)
                time_note = f"\n\n⚠️ Time auto-corrected to 24h format:\n{phone_time.strftime('%H:%M:%S')} → {best_phone_time.strftime('%H:%M:%S')}"
            else:
                time_note = ""
            
            # Find the closest actual data point in the CSV
            start_time, end_time = csv_ranges[best_match]
            try:
                csv_path = self.runs_path / best_match
                csv_df = pd.read_csv(csv_path, usecols=['wall_time_iso'])
                csv_df['wall_time_iso'] = pd.to_datetime(csv_df['wall_time_iso'])
                
                # Build full phone timestamp
                if best_phone_time.year == 1900:
                    phone_time_full = best_phone_time.replace(
                        year=start_time.year,
                        month=start_time.month,
                        day=start_time.day
                    )
                else:
                    phone_time_full = best_phone_time
                
                phone_ts = pd.Timestamp(phone_time_full)
                
                # Find closest row
                time_diffs = abs(csv_df['wall_time_iso'] - phone_ts)
                closest_idx = time_diffs.idxmin()
                closest_time = csv_df['wall_time_iso'].iloc[closest_idx]
                diff_ms = time_diffs.iloc[closest_idx].total_seconds() * 1000
                
                # Check if phone time is before/after CSV range
                phone_before_csv = phone_ts < csv_df['wall_time_iso'].iloc[0]
                phone_after_csv = phone_ts > csv_df['wall_time_iso'].iloc[-1]
                
                if diff_ms < 1:
                    match_info = f"🎯 EXACT MATCH!\nClosest reading: {closest_time.strftime('%H:%M:%S.%f')[:-3]}"
                elif diff_ms < 50:
                    match_info = f"✓ Excellent match ({diff_ms:.1f}ms difference)\nClosest reading: {closest_time.strftime('%H:%M:%S.%f')[:-3]}"
                elif diff_ms < 100:
                    match_info = f"✓ Good match ({diff_ms:.1f}ms difference)\nClosest reading: {closest_time.strftime('%H:%M:%S.%f')[:-3]}"
                elif phone_before_csv:
                    # Phone time is before CSV started - this is okay, auto-trim will handle it
                    csv_start = csv_df['wall_time_iso'].iloc[0]
                    match_info = f"✓ Phone time is {diff_ms:.0f}ms BEFORE sensor started\n"
                    match_info += f"CSV starts at: {csv_start.strftime('%H:%M:%S.%f')[:-3]}\n"
                    match_info += f"(Auto-trim will adjust when you save)"
                elif phone_after_csv:
                    # Phone time is after CSV ended
                    csv_end = csv_df['wall_time_iso'].iloc[-1]
                    match_info = f"⚠️ Phone time is {diff_ms:.0f}ms AFTER sensor ended\n"
                    match_info += f"CSV ends at: {csv_end.strftime('%H:%M:%S.%f')[:-3]}"
                else:
                    match_info = f"✓ {diff_ms:.1f}ms to closest reading\nClosest: {closest_time.strftime('%H:%M:%S.%f')[:-3]}"
            except Exception as e:
                match_info = f"(Could not check closest match: {e})"
                    
            # Show the time range of the matched CSV
            if best_score == 0:
                messagebox.showinfo(
                    "CSV Found", 
                    f"Found matching CSV:\n{best_match}\n\n"
                    f"Time range:\n{start_time.strftime('%Y-%m-%d %H:%M:%S')} to\n"
                    f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"{match_info}{time_note}"
                )
            else:
                messagebox.showinfo(
                    "CSV Found", 
                    f"Best matching CSV:\n{best_match}\n\n"
                    f"Time range:\n{start_time.strftime('%Y-%m-%d %H:%M:%S')} to\n"
                    f"{end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"{match_info}{time_note}"
                )
            
            return best_match
        else:
            self.auto_csv_label.config(
                text="✗ No matching CSV found", 
                foreground="red"
            )
            messagebox.showwarning("Warning", "Could not find a matching CSV file.")
            return None
        
    def update_status(self):
        """Update the status text."""
        self.status_text.config(state="normal")
        self.status_text.delete(1.0, tk.END)
        
        total_videos = self.video_listbox.size()
        configured = len(self.sync_configs)
        
        status = f"Progress: {configured}/{total_videos} videos configured\n\n"
        
        if self.sync_configs:
            status += "Configured videos:\n"
            for name in sorted(self.sync_configs.keys()):
                cfg = self.sync_configs[name]
                csv_name = Path(cfg.csv_path).name
                frames = cfg.trim_end_frame - cfg.trim_start_frame + 1
                status += f"  • {name[:20]}...\n"
                status += f"    CSV: {csv_name[:25]}\n"
                status += f"    Frames: {frames}\n"
                
        self.status_text.insert(1.0, status)
        self.status_text.config(state="disabled")
        
    def save_configs(self):
        """Save all configurations to file."""
        data = {name: config.to_dict() for name, config in self.sync_configs.items()}
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Saved", "Configurations saved successfully!")
        
    def load_configs(self):
        """Load configurations from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                self.sync_configs = {
                    name: VideoSyncConfig.from_dict(cfg) 
                    for name, cfg in data.items()
                }
            except Exception as e:
                print(f"Warning: Could not load configs: {e}")
                self.sync_configs = {}
                
    def create_dataset(self):
        """Create the final synchronized dataset."""
        if not self.sync_configs:
            messagebox.showwarning("Warning", "No videos configured. Please configure at least one video.")
            return
            
        # Ask for output location
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Dataset",
            initialdir=str(self.base_path)
        )
        
        if not output_dir:
            return
            
        output_path = Path(output_dir)
        frames_dir = output_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Creating Dataset...")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        
        progress_label = ttk.Label(progress_window, text="Initializing...")
        progress_label.pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, length=350, mode='determinate')
        progress_bar.pack(pady=10)
        
        self.root.update()
        
        try:
            all_data = []
            total_configs = len(self.sync_configs)
            
            for idx, (video_name, config) in enumerate(self.sync_configs.items()):
                progress_label.config(text=f"Processing: {video_name}")
                progress_bar['value'] = (idx / total_configs) * 100
                self.root.update()
                
                # Load CSV data
                csv_df = pd.read_csv(config.csv_path)
                csv_df['wall_time_iso'] = pd.to_datetime(csv_df['wall_time_iso'])
                
                # Get sync time from phone
                sync_time = datetime.fromisoformat(config.phone_time)
                
                # If sync_time has no date, extract date from CSV filename or first row
                if sync_time.year == 1900:  # Default year from time-only parsing
                    first_csv_time = csv_df['wall_time_iso'].iloc[0]
                    sync_time = sync_time.replace(
                        year=first_csv_time.year,
                        month=first_csv_time.month,
                        day=first_csv_time.day
                    )
                
                # Open video
                cap = cv2.VideoCapture(config.video_path)
                fps = config.video_fps
                
                # Process frames
                for frame_num in range(config.trim_start_frame, config.trim_end_frame + 1):
                    # Calculate the real-world time for this frame
                    frame_offset_seconds = (frame_num - config.sync_frame) / fps
                    frame_time = sync_time + timedelta(seconds=frame_offset_seconds)
                    
                    # Find closest CSV row (sensor data is typically sampled every ~30ms)
                    # We find the absolute closest match, no matter how small the difference
                    time_diffs = abs(csv_df['wall_time_iso'] - frame_time)
                    closest_idx = time_diffs.idxmin()
                    time_diff_ms = time_diffs[closest_idx].total_seconds() * 1000
                    
                    # Skip frames that are outside the CSV time range (> 50ms from any reading)
                    # This handles cases where video extends beyond sensor recording
                    if time_diff_ms > 50:
                        continue
                    
                    # Extract frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Save frame
                    frame_filename = f"{Path(video_name).stem}_frame_{frame_num:06d}.jpg"
                    frame_path = frames_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    
                    # Get sensor data
                    sensor_row = csv_df.iloc[closest_idx].to_dict()
                    sensor_row['frame_path'] = str(frame_path.relative_to(output_path))
                    sensor_row['source_video'] = video_name
                    sensor_row['frame_number'] = frame_num
                    sensor_row['sync_time_diff_ms'] = time_diff_ms
                    
                    all_data.append(sensor_row)
                    
                cap.release()
                
            # Create combined CSV
            progress_label.config(text="Saving dataset CSV...")
            self.root.update()
            
            if all_data:
                df = pd.DataFrame(all_data)
                
                # Reorder columns - put frame info first
                priority_cols = ['frame_path', 'source_video', 'frame_number', 'sync_time_diff_ms', 'wall_time_iso']
                other_cols = [c for c in df.columns if c not in priority_cols]
                df = df[priority_cols + other_cols]
                
                output_csv = output_path / "synchronized_dataset.csv"
                df.to_csv(output_csv, index=False)
                
                progress_window.destroy()
                
                messagebox.showinfo(
                    "Success", 
                    f"Dataset created successfully!\n\n"
                    f"• Frames: {len(all_data)}\n"
                    f"• Videos processed: {total_configs}\n"
                    f"• Output: {output_path}\n\n"
                    f"Files:\n"
                    f"• synchronized_dataset.csv\n"
                    f"• frames/ (contains all extracted frames)"
                )
            else:
                progress_window.destroy()
                messagebox.showwarning("Warning", "No frames were extracted. Check your sync times.")
                
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Error", f"Failed to create dataset: {e}")
            raise
            
    def on_closing(self):
        """Handle window closing."""
        self.is_playing = False
        if self.video_player:
            self.video_player.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DatasetBuilderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

