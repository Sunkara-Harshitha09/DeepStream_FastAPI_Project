import cv2
import json
import os
from collections import defaultdict
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = r".\videos\people_activity_30s.mp4"

MODEL_PATH = r".\yolo11n.pt"

TRACKER_PATH = r".\yolo\trackers\bytetrack_custom.yaml"

OUTPUT_ROOT = r".\person_tracking_output"

FRAME_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "frames"
)

SNAPSHOT_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "snapshots"
)

VIDEO_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "annotated_video"
)

JSON_OUTPUT_DIR = os.path.join(
    OUTPUT_ROOT,
    "json"
)

JSON_OUTPUT_PATH = os.path.join(
    JSON_OUTPUT_DIR,
    "person_tracking.json"
)

ANNOTATED_VIDEO_PATH = os.path.join(
    VIDEO_OUTPUT_DIR,
    "people_tracking_annotated.mp4"
)


# ============================================================
# YOLO SETTINGS
# ============================================================

CONFIDENCE = 0.15
IOU = 0.50
IMAGE_SIZE = 1280

# Save a snapshot every N frames
SNAPSHOT_INTERVAL = 50


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 80)
print("                 PERSON TRACKING STARTED")
print("=" * 80)

print()
print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model: {MODEL_PATH}")
print(f"Tracker: {TRACKER_PATH}")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

if fps <= 0:
    fps = 30.0

duration = total_frames / fps


print()
print("=" * 80)
print("VIDEO INFORMATION")
print("=" * 80)

print(f"Video          : {VIDEO_PATH}")
print(f"Resolution     : {width} x {height}")
print(f"FPS            : {fps:.2f}")
print(f"Total frames   : {total_frames}")
print(f"Duration       : {duration:.2f} seconds")

print()
print("=" * 80)
print("TRACKING SETTINGS")
print("=" * 80)

print(f"Confidence     : {CONFIDENCE}")
print(f"IoU            : {IOU}")
print(f"Image size     : {IMAGE_SIZE}")
print(f"Tracker        : {TRACKER_PATH}")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    ANNOTATED_VIDEO_PATH,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not create annotated output video."
    )


# ============================================================
# TRACKING DATA
# ============================================================

frame_records = []

unique_track_ids = set()

track_history = defaultdict(list)

max_people_in_frame = 0

total_person_detections = 0

frame_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    timestamp_seconds = (
        frame_number - 1
    ) / fps

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        iou=IOU,
        imgsz=IMAGE_SIZE,
        tracker=TRACKER_PATH,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    current_frame_people = []

    # --------------------------------------------------------
    # READ TRACKING RESULTS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()

        confidences = boxes.conf.cpu().numpy()

        class_ids = boxes.cls.cpu().numpy()

        if boxes.id is not None:

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .numpy()
            )

        else:

            track_ids = []

        for index, box in enumerate(xyxy):

            class_id = int(
                class_ids[index]
            )

            # Only person class
            if class_id != 0:
                continue

            confidence = float(
                confidences[index]
            )

            x1, y1, x2, y2 = map(
                float,
                box
            )

            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            if index < len(track_ids):

                track_id = int(
                    track_ids[index]
                )

                unique_track_ids.add(
                    track_id
                )

            else:

                track_id = None

            # ------------------------------------------------
            # CENTER POINT
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            person_data = {
                "track_id": track_id,
                "confidence": round(
                    confidence,
                    5
                ),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                },
                "center": {
                    "x": round(
                        center_x,
                        2
                    ),
                    "y": round(
                        center_y,
                        2
                    )
                }
            }

            current_frame_people.append(
                person_data
            )

            total_person_detections += 1

            # ------------------------------------------------
            # TRACK HISTORY
            # ------------------------------------------------

            if track_id is not None:

                track_history[
                    track_id
                ].append(
                    {
                        "frame": frame_number,
                        "timestamp_seconds": round(
                            timestamp_seconds,
                            3
                        ),
                        "center_x": round(
                            center_x,
                            2
                        ),
                        "center_y": round(
                            center_y,
                            2
                        ),
                        "confidence": round(
                            confidence,
                            5
                        ),
                        "bbox": {
                            "x1": round(
                                x1,
                                2
                            ),
                            "y1": round(
                                y1,
                                2
                            ),
                            "x2": round(
                                x2,
                                2
                            ),
                            "y2": round(
                                y2,
                                2
                            )
                        }
                    }
                )

    # --------------------------------------------------------
    # CURRENT FRAME COUNT
    # --------------------------------------------------------

    people_count = len(
        current_frame_people
    )

    if people_count > max_people_in_frame:

        max_people_in_frame = (
            people_count
        )

    # --------------------------------------------------------
    # FRAME RECORD
    # --------------------------------------------------------

    frame_record = {
        "frame_number": frame_number,
        "timestamp_seconds": round(
            timestamp_seconds,
            3
        ),
        "person_count": people_count,
        "persons": current_frame_people
    }

    frame_records.append(
        frame_record
    )

    # --------------------------------------------------------
    # SAVE ANNOTATED FRAME
    # --------------------------------------------------------

    frame_filename = (
        f"frame_{frame_number:06d}.jpg"
    )

    frame_path = os.path.join(
        FRAME_OUTPUT_DIR,
        frame_filename
    )

    cv2.imwrite(
        frame_path,
        annotated_frame
    )

    # --------------------------------------------------------
    # SAVE SNAPSHOT
    # --------------------------------------------------------

    if (
        frame_number == 1
        or frame_number % SNAPSHOT_INTERVAL == 0
        or frame_number == total_frames
    ):

        snapshot_filename = (
            f"snapshot_{frame_number:06d}.jpg"
        )

        snapshot_path = os.path.join(
            SNAPSHOT_OUTPUT_DIR,
            snapshot_filename
        )

        cv2.imwrite(
            snapshot_path,
            annotated_frame
        )

    # --------------------------------------------------------
    # WRITE VIDEO FRAME
    # --------------------------------------------------------

    writer.write(
        annotated_frame
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        frame_number % 25 == 0
        or frame_number == total_frames
    ):

        print(
            f"Processed "
            f"{frame_number}/{total_frames} "
            f"| people: {people_count} "
            f"| unique IDs: "
            f"{len(unique_track_ids)}"
        )


# ============================================================
# RELEASE RESOURCES
# ============================================================

cap.release()
writer.release()


# ============================================================
# BUILD TRACK SUMMARY
# ============================================================

tracks_summary = []

for track_id in sorted(
    track_history.keys()
):

    history = track_history[
        track_id
    ]

    if not history:
        continue

    first_detection = history[0]

    last_detection = history[-1]

    track_summary = {

        "track_id": track_id,

        "first_seen": {
            "frame": first_detection[
                "frame"
            ],
            "timestamp_seconds":
                first_detection[
                    "timestamp_seconds"
                ]
        },

        "last_seen": {
            "frame": last_detection[
                "frame"
            ],
            "timestamp_seconds":
                last_detection[
                    "timestamp_seconds"
                ]
        },

        "observed_duration_seconds":
            round(
                last_detection[
                    "timestamp_seconds"
                ]
                -
                first_detection[
                    "timestamp_seconds"
                ],
                3
            ),

        "frames_tracked": len(
            history
        ),

        "positions": history

    }

    tracks_summary.append(
        track_summary
    )


# ============================================================
# FINAL JSON
# ============================================================

tracking_output = {

    "video_name": os.path.basename(
        VIDEO_PATH
    ),

    "video_information": {

        "width": width,

        "height": height,

        "fps": round(
            fps,
            3
        ),

        "total_frames": total_frames,

        "duration_seconds":
            round(
                duration,
                3
            )
    },

    "tracking_configuration": {

        "model": MODEL_PATH,

        "tracker": TRACKER_PATH,

        "confidence":
            CONFIDENCE,

        "iou": IOU,

        "image_size":
            IMAGE_SIZE
    },

    "summary": {

        "maximum_people_in_one_frame":
            max_people_in_frame,

        "unique_tracking_ids":
            len(unique_track_ids),

        "total_person_detections":
            total_person_detections
    },

    "tracks": tracks_summary,

    "frames": frame_records
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    JSON_OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        tracking_output,
        file,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("                 TRACKING COMPLETE")
print("=" * 80)

print()
print(
    f"Frames processed          : "
    f"{frame_number}"
)

print(
    f"Maximum people in frame  : "
    f"{max_people_in_frame}"
)

print(
    f"Unique tracking IDs       : "
    f"{len(unique_track_ids)}"
)

print(
    f"Total person detections   : "
    f"{total_person_detections}"
)

print()
print("OUTPUTS")
print("-" * 80)

print(
    f"Frames      : "
    f"{FRAME_OUTPUT_DIR}"
)

print(
    f"Snapshots   : "
    f"{SNAPSHOT_OUTPUT_DIR}"
)

print(
    f"Video       : "
    f"{ANNOTATED_VIDEO_PATH}"
)

print(
    f"JSON        : "
    f"{JSON_OUTPUT_PATH}"
)

print()
print("=" * 80)