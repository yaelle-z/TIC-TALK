#!/usr/bin/env python3
import os
import csv
import argparse
from ultralytics import YOLO


def parse_metadata_from_filename(file_name: str):
    
    base = os.path.splitext(file_name)[0]
    author = ""
    show_name = ""
    timecode = ""

    if "_frame" in base:
        prefix, frame_part = base.split("_frame", 1)
        timecode = frame_part
    else:
        prefix = base

    parts = prefix.split("_")
    if len(parts) >= 2:
        author = parts[0]
        show_name = "_".join(parts[1:])
    elif len(parts) == 1:
        author = parts[0]

    return author, show_name, timecode


def get_class_name(class_names, class_id: int):
    """
    class_names can be a  dict {0: 'xxx'} or a list ['xxx', ...].
    """
    if isinstance(class_names, dict):
        return class_names.get(class_id, str(class_id))
    else:
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return str(class_id)


def process_show(show_dir, output_root, model, class_names, frame_step: int):
    image_files = sorted(
        f for f in os.listdir(show_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        print(f"[WARNING] No images in {show_dir}")
        return

   
    if frame_step > 1:
        image_files = image_files[::frame_step]

    show_id = os.path.basename(show_dir.rstrip(os.sep))
    csv_path = os.path.join(output_root, f"{show_id}_plans_detectes.csv")

    if os.path.exists(csv_path):
        print(f"[SKIP] {show_id} : {csv_path} already exists")
        return

    print(f"[START] {show_id} — {len(image_files)} images (after sampling)")
    print(f"      → output: {csv_path}")

    os.makedirs(output_root, exist_ok=True)

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        
        header = [
            "file_name",
            "author",
            "show_name",
            "timecode",
            
            "plan_label",
            "plan_id",
            "plan_score",
        ]
        writer.writerow(header)

        n = len(image_files)

        for idx, file_name in enumerate(image_files):
            image_path = os.path.join(show_dir, file_name)

            author, show_name, timecode = parse_metadata_from_filename(file_name)
            row = [file_name, author, show_name, timecode]

            try:
                results = model(image_path, verbose=False)
            except Exception as e:
                print(f"[ERROR] {show_id}: erreur sur {file_name} → {e}")
                row.extend(["", "", ""])
                writer.writerow(row)
                continue

            result = results[0]

            if result.probs is None:
                # no prediction (should be uncommon)
                row.extend(["", "", ""])
            else:
                probs = result.probs
                class_id = int(probs.top1)
                try:
                    score = float(probs.top1conf.item())
                except AttributeError:
                    score = float(probs.data[class_id].item())

                label = get_class_name(class_names, class_id)

                row.extend([label, class_id, score])

            writer.writerow(row)

            if (idx + 1) % 1000 == 0:
                print(f"  {show_id}: {idx + 1}/{n} images traitées")

    print(f"[DONE] {show_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply YOLOv8 shot-type classifier to stand-up frames."
    )
    parser.add_argument(
        "--root-folder",
        required=True,
        help="Root folder with one subfolder per show, each containing images.",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder where per-show CSV files will be written.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help=(
            "Path to YOLOv8 classification weights "
            "(e.g. runs/classify/train/weights/best.pt)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=8,
        help="Keep 1 image every N frames (8 si tes frames sont à ~8 fps).",
    )

    args = parser.parse_args()

    print(f"[MODEL] Loading YOLO cls model from {args.model_path}")
    model = YOLO(args.model_path)

    if args.device:
        try:
            model.to(args.device)
        except Exception as e:
            print(f"[WARNING] Could not move model to device '{args.device}': {e}")
            print("          Continuing with default device.")

    class_names = model.names  

    root_folder = args.root_folder
    output_root = args.output_folder

    for show_name in sorted(os.listdir(root_folder)):
        show_dir = os.path.join(root_folder, show_name)
        if os.path.isdir(show_dir):
            process_show(
                show_dir=show_dir,
                output_root=output_root,
                model=model,
                class_names=class_names,
                frame_step=args.frame_step,
            )


if __name__ == "__main__":
    main()
