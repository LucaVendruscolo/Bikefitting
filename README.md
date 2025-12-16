Bikefitting

Folders

- `create_labeled_dataset/`  
  GUI tool to sync videos + IMU runs and export a labeled dataset (`output/synchronized_dataset.csv` + `output/frames/`).
- `bike_angle_detection_model/`  
  Trains and runs the bike angle model (mask bikes with YOLO segmentation, then train classifier).
- `joint_angle_detection/`  
  Live webcam joint detection + basic joint angles using YOLOv8 pose.
- `sam3/`  
  External package (installed editable).

Setup

You need Python 3.8+ (GPU is optional).

Mac/Linux (uses install.sh)

```bash
conda create -n bikefitting python=3.12
conda activate bikefitting
./install.sh
```

CPU-only install.sh:

```bash
BIKEFITTING_CPU_ONLY=1 ./install.sh
```

Windows (manual)

```bash
python -m venv bikeEnv
bikeEnv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .\sam3 
```

Quick start

- Build dataset: read `create_labeled_dataset/README.md`
- Train bike angle: read `bike_angle_detection_model/README.md`
- Live joint angles: read `joint_angle_detection/README.md`
```