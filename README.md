# CodeAlpha_ObjectDetection

**Real-time Object Detection & Tracking using YOLOv8 + ByteTrack**

Built for the [CodeAlpha](https://www.codealpha.tech) AI Internship — Task 4.

---

## Features

| Feature | Detail |
|---|---|
| Detection model | YOLOv8n (nano) — swap to `yolov8s/m/l/x.pt` for more accuracy |
| Tracker | ByteTrack (built into Ultralytics, no extra install) |
| Input sources | Webcam (live) or any video file |
| Bounding boxes | Colour-coded per class |
| Labels | Class name + confidence % + persistent Track ID |
| HUD | Live FPS + object count + frame number |
| Output | Auto-saved MP4 to the `output/` folder |
| Controls | Q=Quit, P=Pause/Resume, S=Screenshot |

---

## Project Structure

```
CodeAlpha_ObjectDetection/
├── main.py          ← Entry point (run this)
├── detector.py      ← YOLOv8 + ByteTrack wrapper class
├── utils.py         ← Drawing helpers + FPS counter
├── requirements.txt ← All Python dependencies
├── output/          ← Saved output videos go here
└── README.md
```

---

## Setup

### 1 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** install the CUDA build of PyTorch first, then run the above:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

### 3 — First run (model auto-download)

On the very first run, YOLOv8 will automatically download `yolov8n.pt` (~6 MB).  
Internet connection required once only.

---

## Usage

### Webcam (default — camera index 0)

```bash
python main.py --source 0
```

### Video file

```bash
python main.py --source path/to/your/video.mp4
```

### Adjust confidence threshold

```bash
python main.py --source 0 --conf 0.5
```

### Use a larger (more accurate) model

```bash
python main.py --source demo.mp4 --weights yolov8m.pt
```

### Headless mode (no display window — for servers)

```bash
python main.py --source demo.mp4 --no-display
```

### Don't save output video

```bash
python main.py --source 0 --no-save
```

### Custom output directory

```bash
python main.py --source demo.mp4 --output-dir my_results
```

### All options

```
--source      Source: '0' for webcam, or path to video file (default: 0)
--weights     YOLOv8 weights file (default: yolov8n.pt)
--conf        Confidence threshold 0–1 (default: 0.30)
--iou         NMS IoU threshold 0–1 (default: 0.50)
--output-dir  Folder for saved videos (default: output/)
--no-display  Disable preview window
--no-save     Do not save output video
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit |
| `P` | Pause / Resume |
| `S` | Save screenshot of current frame |

---

## Model Options

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| yolov8n.pt | ~6 MB | ⚡ Fastest | Good |
| yolov8s.pt | ~22 MB | Fast | Better |
| yolov8m.pt | ~50 MB | Medium | Great |
| yolov8l.pt | ~87 MB | Slow | Excellent |
| yolov8x.pt | ~131 MB | Slowest | Best |

---

## Output

Processed videos are saved automatically to `output/` with this naming pattern:

```
output/tracked_<source>_<timestamp>.mp4
```

---

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection + ByteTrack
- [OpenCV](https://opencv.org) — video I/O, display, drawing
- [PyTorch](https://pytorch.org) — inference backend

---

*CodeAlpha AI Internship — Task 4: Object Detection and Tracking*
