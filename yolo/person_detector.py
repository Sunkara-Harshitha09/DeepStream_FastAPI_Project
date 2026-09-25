import cv2
import json
import os

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = r".\videos\people_activity_30s.mp4"
MODEL_PATH = r".\yolo11n.pt"

OUTPUT_ROOT = r".\person_detection_output"

FRAMES_DIR = os.path.join(
    OUTPUT_ROOT,
    "frames"
)

SNAPSHOTS_DIR = os.path.join(
    OUTPUT_ROOT,
    "snapshots"
)

ANNOTATED_DIR = os.path.join(
    OUTPUT_ROOT,
    "annotated_video"
)

JSON_DIR = os.path.join(
    OUTPUT_ROOT,
    "json"
)

ANNOTATED_VIDEO = os.path.join(
    ANNOTATED_DIR,
    "people_detection_annotated.mp4"
)

DETECTIONS_JSON = os.path.join(
    JSON_DIR,
    "person_detections.json"
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

# Lower confidence helps detect smaller people.
CONFIDENCE = 0.15

# Higher resolution helps detect distant people.
IMAGE_SIZE = 1280

IOU = 0.50

SNAPSHOT_INTERVAL = 50


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("              PERSON DETECTION VERIFICATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not os.path.exists(VIDEO_PATH):

        print(
            f"ERROR: Video not found:\n{VIDEO_PATH}"
        )

        return

    if not os.path.exists(MODEL_PATH):

        print(
            f"ERROR: Model not found:\n{MODEL_PATH}"
        )

        return

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for folder in [
        FRAMES_DIR,
        SNAPSHOTS_DIR,
        ANNOTATED_DIR,
        JSON_DIR
    ]:

        os.makedirs(
            folder,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading YOLO detection model...")

    model = YOLO(
        MODEL_PATH
    )

    print(
        "YOLO detection model loaded successfully."
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )

    print(
        f"\nResolution: {width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"Confidence threshold: {CONFIDENCE}"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    # --------------------------------------------------------
    # Create annotated video
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        ANNOTATED_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_detections = []

    frame_number = 0

    maximum_persons_in_frame = 0

    total_person_detections = 0

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    print("\nProcessing video...")
    print("Saving every annotated frame...\n")

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

        # ----------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------

        results = model.predict(
            frame,
            classes=[0],
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        result = results[0]

        person_count = 0

        # ----------------------------------------------------
        # Process people
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            person_count = len(
                boxes
            )

            maximum_persons_in_frame = max(
                maximum_persons_in_frame,
                person_count
            )

            total_person_detections += (
                person_count
            )

            for index, bbox in enumerate(
                boxes
            ):

                x1, y1, x2, y2 = bbox

                confidence = float(
                    confidences[index]
                )

                # ------------------------------------------------
                # Store detection
                # ------------------------------------------------

                all_detections.append(
                    {
                        "frame_number":
                            frame_number,

                        "timestamp_seconds":
                            round(
                                timestamp,
                                3
                            ),

                        "person_number_in_frame":
                            index + 1,

                        "confidence":
                            round(
                                confidence,
                                4
                            ),

                        "bounding_box":
                            {
                                "x1":
                                    round(
                                        float(x1),
                                        2
                                    ),

                                "y1":
                                    round(
                                        float(y1),
                                        2
                                    ),

                                "x2":
                                    round(
                                        float(x2),
                                        2
                                    ),

                                "y2":
                                    round(
                                        float(y2),
                                        2
                                    )
                            }
                    }
                )

                # ------------------------------------------------
                # Draw bounding box
                # ------------------------------------------------

                x1_int = int(x1)
                y1_int = int(y1)
                x2_int = int(x2)
                y2_int = int(y2)

                cv2.rectangle(
                    frame,
                    (x1_int, y1_int),
                    (x2_int, y2_int),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"PERSON "
                    f"{index + 1} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1_int,
                        max(
                            25,
                            y1_int - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"Frame: "
                f"{frame_number}/{total_frames}"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"Time: "
                f"{timestamp:.2f}s"
            ),
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                f"Persons detected: "
                f"{person_count}"
            ),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Save every frame
        # ----------------------------------------------------

        frame_path = os.path.join(
            FRAMES_DIR,
            f"frame_{frame_number:06d}.jpg"
        )

        cv2.imwrite(
            frame_path,
            frame
        )

        # ----------------------------------------------------
        # Save snapshots
        # ----------------------------------------------------

        if (
            frame_number == 1
            or frame_number == total_frames
            or frame_number % SNAPSHOT_INTERVAL == 0
        ):

            snapshot_path = os.path.join(
                SNAPSHOTS_DIR,
                f"snapshot_{frame_number:06d}.jpg"
            )

            cv2.imwrite(
                snapshot_path,
                frame
            )

        # ----------------------------------------------------
        # Save annotated video frame
        # ----------------------------------------------------

        writer.write(
            frame
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_number % 25 == 0:

            print(
                f"Processed "
                f"{frame_number}/{total_frames}"
                f" | persons in frame: "
                f"{person_count}"
            )

    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    writer.release()

    # ========================================================
    # SAVE JSON
    # ========================================================

    output = {

        "video_name":
            os.path.basename(
                VIDEO_PATH
            ),

        "video_duration_seconds":
            round(
                duration,
                3
            ),

        "total_frames":
            total_frames,

        "maximum_persons_detected_in_one_frame":
            maximum_persons_in_frame,

        "total_person_detections":
            total_person_detections,

        "confidence_threshold":
            CONFIDENCE,

        "image_size":
            IMAGE_SIZE,

        "detections":
            all_detections
    }

    with open(
        DETECTIONS_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print("             PERSON DETECTION COMPLETED")
    print("=" * 75)

    print(
        f"Frames processed: "
        f"{total_frames}"
    )

    print(
        f"Maximum persons in one frame: "
        f"{maximum_persons_in_frame}"
    )

    print(
        f"Total person detections: "
        f"{total_person_detections}"
    )

    print(
        f"\nAnnotated video:"
        f"\n{ANNOTATED_VIDEO}"
    )

    print(
        f"\nEvery frame:"
        f"\n{FRAMES_DIR}"
    )

    print(
        f"\nSnapshots:"
        f"\n{SNAPSHOTS_DIR}"
    )

    print(
        f"\nDetection JSON:"
        f"\n{DETECTIONS_JSON}"
    )

    print("=" * 75)


if __name__ == "__main__":

    main()