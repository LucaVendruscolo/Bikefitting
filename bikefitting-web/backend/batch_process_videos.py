import os
import sys
import glob

# Ensure we can import from the processing module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing.video_processor import VideoProcessor

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    input_folder = "C:/Users/fivos/Videos/bike-videos"   #  source folder
    output_folder = "processed_results"                  # output folder
    model_path = "../../models/best_model.pt"            #  model path
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ==========================================
    # 2. INITIALIZE PROCESSOR
    # ==========================================
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Initializing Processor...")
    try:
        processor = VideoProcessor(model_path=model_path)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # ==========================================
    # 3. BATCH PROCESS
    # ==========================================
    # Find all MP4 files
    mp4_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    mov_files = glob.glob(os.path.join(input_folder, "*.mov"))
    
    # Combine the lists
    video_files = mp4_files + mov_files
    
    if not video_files:
        print(f"No .mp4 or .mov videos found in {input_folder}")
        return

    print(f"Found {len(video_files)} videos. Starting batch...")

    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        print(f"\n[{i+1}/{len(video_files)}] Processing: {filename}")
        
        # Define unique output paths for this specific video
        save_video_path = os.path.join(output_folder, f"{name_no_ext}_processed.mp4")
        save_csv_path = os.path.join(output_folder, f"{name_no_ext}_data.csv")
        
        try:
            # Pass the unique 'save_csv_path' to your updated function
            processor.process_video(
                input_path=video_path,
                output_path=save_video_path,
                csv_path=save_csv_path,  # <--- PASSING THE PATH HERE
                output_fps=30
            )
            
            print(f"   -> Video saved to: {save_video_path}")
            print(f"   -> Data saved to:  {save_csv_path}")

        except Exception as e:
            print(f"   -> Error processing {filename}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()