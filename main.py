"""
main.py
-------
Entry point for YOLOv8 + ByteTrack Object Detection & Tracking.

Supports:
  • Webcam live feed   →  python main.py --source 0
  • Any video file     →  python main.py --source path/to/video.mp4
  • Headless (no GUI)  →  python main.py --source video.mp4 --no-display
  • Skip saving        →  python main.py --source 0 --no-save

Keyboard shortcuts (when window is open):
  q / ESC  — quit
  p        — pause / resume
  s        — save screenshot of current frame
"""

import argparse
import os
import sys
import time
import cv2
from datetime import datetime
from pathlib import Path

from detector import ObjectDetector
from utils import draw_detection, draw_fps, draw_info_bar, FPSCounter


# ── CLI ─────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YOLOv8 + ByteTrack — Object Detection & Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Webcam
  python main.py --source 0

  # Video file
  python main.py --source demo.mp4

  # Higher confidence, larger model
  python main.py --source demo.mp4 --conf 0.5 --weights yolov8m.pt

  # Headless (server / no monitor)
  python main.py --source demo.mp4 --no-display
        """,
    )
    p.add_argument("--source",     type=str,   default="0",
                   help="'0' for webcam, or path to a video file (default: 0)")
    p.add_argument("--weights",    type=str,   default="yolov8n.pt",
                   help="YOLOv8 weights (default: yolov8n.pt — auto-downloaded)")
    p.add_argument("--conf",       type=float, default=0.30,
                   help="Detection confidence threshold (default: 0.30)")
    p.add_argument("--iou",        type=float, default=0.50,
                   help="NMS IoU threshold (default: 0.50)")
    p.add_argument("--output-dir", type=str,   default="output",
                   help="Folder for saved output videos (default: output/)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable live preview window (headless mode)")
    p.add_argument("--no-save",    action="store_true",
                   help="Do not save the processed video")
    return p


# ── Source handling ──────────────────────────────────────────────────────────────

def resolve_source(source_str: str) -> tuple:
    """
    Returns (cv2_source, display_label).
    cv2_source is an int (webcam index) or str (file path).
    """
    # Webcam?
    try:
        idx = int(source_str)
        return idx, f"Webcam[{idx}]"
    except ValueError:
        pass

    # Video file
    path = Path(source_str)
    if not path.exists():
        print(f"[ERROR] File not found: {source_str}")
        sys.exit(1)

    return str(path), path.name


# ── Video writer ─────────────────────────────────────────────────────────────────

def make_writer(
    output_dir: str,
    source_label: str,
    cap: cv2.VideoCapture,
) -> tuple:
    """
    Create a VideoWriter and return (writer, output_path).
    Returns (None, None) if creation fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe  = "".join(c if c.isalnum() or c in "-_." else "_" for c in source_label)
    fname = f"tracked_{safe}_{ts}.mp4"
    out   = os.path.join(output_dir, fname)

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0          # sensible default for webcam

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (W, H))

    if not writer.isOpened():
        print(f"[WARN] Could not open VideoWriter at {out}; output will NOT be saved.")
        return None, None

    print(f"[Writer] Output → {out}  ({W}×{H} @ {fps:.0f} fps)")
    return writer, out


# ── Main loop ────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = build_parser().parse_args()
    source, source_label = resolve_source(args.source)

    # ── Load model ────────────────────────────────────────────────────────────
    detector = ObjectDetector(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
    )

    # ── Open capture ──────────────────────────────────────────────────────────
    print(f"\n[Capture] Opening: {source_label}")
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        print(f"[Capture] Total frames in file: {total_frames}")

    # ── Video writer ──────────────────────────────────────────────────────────
    writer     = None
    output_path = None
    if not args.no_save:
        writer, output_path = make_writer(args.output_dir, source_label, cap)

    # ── Window ────────────────────────────────────────────────────────────────
    WIN = "YOLOv8 + ByteTrack  |  Q=Quit  P=Pause  S=Screenshot"
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)

    # ── Processing loop ───────────────────────────────────────────────────────
    fps_ctr   = FPSCounter(window=30)
    frame_num = 0
    paused    = False
    frame     = None                    # hold last frame for pause display

    print(f"\n[Running]  Press Q/ESC to quit | P to pause | S for screenshot\n")

    try:
        while True:
            # ── Pause: just re-show last frame, handle keys ────────────────
            if paused:
                if not args.no_display and frame is not None:
                    cv2.imshow(WIN, frame)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[Quit] User pressed Q/ESC.")
                    break
                elif key == ord("p"):
                    paused = False
                    print("[Resumed]")
                continue

            # ── Read frame ────────────────────────────────────────────────
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480),interpolation=cv2.INTER_LINEAR)
            if not ret:
                if isinstance(source, str):
                    print("\n[Done] End of video file.")
                else:
                    print("\n[WARN] Webcam returned empty frame — retrying…")
                    time.sleep(0.05)
                    continue
                break

            frame_num  += 1
            fps_ctr.tick()

            # ── Detect + track ────────────────────────────────────────────
            frame = cv2.resize(frame, (640, 480))
            detections = detector.track(frame)

            # ── Draw detections ───────────────────────────────────────────
            for det in detections:
                draw_detection(frame, det)

            # ── HUD ───────────────────────────────────────────────────────
            draw_fps(frame, fps_ctr.fps)
            draw_info_bar(frame, len(detections), source_label, frame_num)

            # ── Save ──────────────────────────────────────────────────────
            if writer is not None:
                writer.write(frame)

            # ── Display ───────────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow(WIN, frame)

            # ── Keys ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):           # Q or ESC
                print("\n[Quit] User pressed Q/ESC.")
                break

            elif key == ord("p"):               # Pause
                paused = True
                print("[Paused]  Press P to resume.")

            elif key == ord("s"):               # Screenshot
                shot = f"screenshot_{frame_num:06d}.jpg"
                cv2.imwrite(shot, frame)
                print(f"[Screenshot] Saved → {shot}")

            # ── Window closed via ✕ button ────────────────────────────────
            if not args.no_display:
                try:
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        print("\n[Quit] Window closed.")
                        break
                except cv2.error:
                    break

            # ── Progress log (every 100 frames for video files) ───────────
            if isinstance(source, str) and frame_num % 100 == 0 and total_frames > 0:
                pct = frame_num / total_frames * 100
                print(
                    f"[Progress] {frame_num:6d}/{total_frames}  "
                    f"({pct:5.1f}%)  FPS:{fps_ctr.fps:5.1f}  "
                    f"Dets:{len(detections)}"
                )

    except KeyboardInterrupt:
        print("\n[Interrupted] Ctrl+C received.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n[Saved] Output video → {output_path}")
        cv2.destroyAllWindows()
        print(f"[Done]  Processed {frame_num} frames.")


if __name__ == "__main__":
    main()
