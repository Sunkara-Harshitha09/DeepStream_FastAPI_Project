import cv2
import json
import os
from ultralytics import YOLO


VIDEO_PATH = r".\videos\people_activity_30s.mp4"
MODEL_PATH = r".\yolo11n-pose.pt"

OUTPUT_JSON = r".\json_output\all_persons.json"


def main():

    print("=" * 70)
    print("          ALL PERSON DETECTION")
    print("=" * 70)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    # --------------------------------------------------------
    # Store every person seen
    # --------------------------------------------------------

    persons = {}

    frame_number = 0

    max_persons_in_one_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        timestamp = (
            frame_number / fps
            if fps > 0
            else 0
        )

        # Pose detection WITHOUT relying on tracking IDs
        results = model.predict(
            frame,
            classes=[0],
            conf=0.25,
            verbose=False
        )

        if not results:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        confidence_values = (
            result.boxes.conf.cpu().numpy()
        )

        person_count = len(boxes)

        max_persons_in_one_frame = max(
            max_persons_in_one_frame,
            person_count
        )

        # ----------------------------------------------------
        # Store every detection in this frame
        # ----------------------------------------------------

        for detection_index, bbox in enumerate(boxes):

            x1, y1, x2, y2 = bbox

            confidence = float(
                confidence_values[
                    detection_index
                ]
            )

            detection = {
                "frame_number": frame_number,
                "timestamp": round(
                    timestamp,
                    3
                ),
                "confidence": round(
                    confidence,
                    4
                ),
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2)
                }
            }

            # Temporary detection ID
            detection_id = (
                f"frame_{frame_number}"
                f"_person_{detection_index + 1}"
            )

            persons[detection_id] = detection

        if frame_number % 100 == 0:

            print(
                f"Processed "
                f"{frame_number}/{total_frames} "
                f"| persons in current frame: "
                f"{person_count}"
            )

    cap.release()

    # --------------------------------------------------------
    # Create JSON
    # --------------------------------------------------------

    output = {

        "video_name":
            os.path.basename(VIDEO_PATH),

        "video_duration_seconds":
            round(duration, 3),

        "total_frames":
            total_frames,

        "maximum_persons_detected_in_one_frame":
            max_persons_in_one_frame,

        "total_person_detections":
            len(persons),

        "detections":
            list(persons.values())
    }

    os.makedirs(
        os.path.dirname(OUTPUT_JSON),
        exist_ok=True
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("DETECTION COMPLETED")
    print("=" * 70)

    print(
        f"Maximum persons in one frame: "
        f"{max_persons_in_one_frame}"
    )

    print(
        f"Total person detections: "
        f"{len(persons)}"
    )

    print(
        f"JSON saved to:\n{OUTPUT_JSON}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()