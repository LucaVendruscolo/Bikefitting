import os
import sys

# Ensure we can import from the processing module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing.video_processor import VideoProcessor

def main():
    # 1. SETUP PATHS
    # Replace this with the path to your test video
    input_video = "C:/Users/fivos/Videos/bike-videos/my_cycling_video.mp4" 
    
    # Where to save the result
    output_video = "output_test.mp4"
    
    # Path to your trained bike angle model (ConvNeXt)
    # Adjust this relative path if your models are stored elsewhere
    model_path = "../../models/best_model.pt" 

    # Check if files exist
    if not os.path.exists(input_video):
        print(f"Error: Video not found at {input_video}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Note: If you don't have the bike angle model yet, 
        # you might need to comment out the angle_predictor lines 
        # in video_processor.py temporarily to test just the skeleton.
        return

    print("Initializing Processor...")
    # Initialize the processor (this loads YOLO and ConvNeXt)
    try:
        processor = VideoProcessor(model_path=model_path)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    print(f"Processing {input_video}...")
    
    # 2. RUN PROCESSING
    # This runs your new logic: Pose Detection -> Curve Fitting -> Video Generation
    stats = processor.process_video(
        input_path=input_video,
        output_path=output_video,
        output_fps=30,
        max_duration_sec=None,   
        start_time=0
    )
    
    print(f"\nOutput video saved to: {os.path.abspath(output_video)}")

if __name__ == "__main__":
    main()