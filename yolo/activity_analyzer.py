import os
import json
import math
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = r".\videos\people_activity_30s.mp4"

# Main detector - authoritative person detector
DETECTION_MODEL_PATH = r".\yolo11n.pt"

# Pose model - used only for activity classification
POSE_MODEL_PATH = r".\yolo11n-pose.pt"

# Output directory
OUTPUT_ROOT = r".\activity_output"

FRAMES_DIR = os.path.join(OUTPUT_ROOT, "frames")
SNAPSHOTS_DIR = os.path.join(OUTPUT_ROOT, "snapshots")
VIDEO_DIR = os.path.join(OUTPUT_ROOT, "annotated_video")
JSON_DIR = os.path.join(OUTPUT_ROOT, "json")

ANNOTATED_VIDEO_PATH = os.path.join(
    VIDEO_DIR,
    "people_activity_annotated.mp4"
)

JSON_OUTPUT_PATH = os.path.join(
    JSON_DIR,
    "people_activity_30s.json"
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE = 0.15
IOU = 0.50
IMAGE_SIZE = 1280

# Process pose model every N frames.
# 1 = every frame.
POSE_INTERVAL = 1

# Save every annotated frame
SAVE_ALL_FRAMES = True

# Save one snapshot every N frames
SNAPSHOT_INTERVAL = 25


# ============================================================
# TRACKING SETTINGS
# ============================================================

# Maximum frames a disappeared person can remain alive.
MAX_MISSED_FRAMES = 45

# IoU required for matching detections to existing tracks.
TRACK_IOU_THRESHOLD = 0.25

# Center-distance threshold relative to person's box size.
CENTER_DISTANCE_THRESHOLD = 1.5


# ============================================================
# ACTIVITY SETTINGS
# ============================================================

# Number of recent positions used for movement calculation
MOVEMENT_HISTORY = 7

# Movement threshold.
# This is normalized using person bounding-box height.
WALKING_MOVEMENT_THRESHOLD = 0.035

# If movement is very small, classify as standing unless
# pose indicates sitting.
STANDING_MOVEMENT_THRESHOLD = 0.012

# Pose confidence
KEYPOINT_CONFIDENCE = 0.25

# Number of frames required before accepting a track
MIN_TRACK_FRAMES = 5


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_ROOT,
    FRAMES_DIR,
    SNAPSHOTS_DIR,
    VIDEO_DIR,
    JSON_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iou(box_a, box_b):
    """
    Calculate IoU between two bounding boxes.

    Box format:
    [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_height(box):
    return max(1.0, box[3] - box[1])


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    return math.sqrt(
        (ax - bx) ** 2 +
        (ay - by) ** 2
    )


def get_pose_for_person(person_box, pose_boxes, pose_keypoints):
    """
    Find the pose detection belonging to a detector person.
    """

    best_index = -1
    best_iou = 0.0

    for index, pose_box in enumerate(pose_boxes):

        iou = calculate_iou(person_box, pose_box)

        if iou > best_iou:
            best_iou = iou
            best_index = index

    if best_index == -1:
        return None

    # Require reasonable overlap
    if best_iou < 0.10:
        return None

    return pose_keypoints[best_index]


def classify_from_pose(person_box, keypoints):
    """
    Classify sitting / standing using pose geometry.

    Returns:
        sitting
        standing
        None
    """

    if keypoints is None:
        return None

    try:

        # YOLO pose keypoint indexes
        # 5  = left shoulder
        # 6  = right shoulder
        # 11 = left hip
        # 12 = right hip
        # 13 = left knee
        # 14 = right knee
        # 15 = left ankle
        # 16 = right ankle

        required = [5, 6, 11, 12, 13, 14]

        points = []

        for idx in required:

            if idx >= len(keypoints):
                return None

            x, y, confidence = keypoints[idx]

            if confidence < KEYPOINT_CONFIDENCE:
                return None

            points.append((float(x), float(y)))

        left_shoulder = points[0]
        right_shoulder = points[1]

        left_hip = points[2]
        right_hip = points[3]

        left_knee = points[4]
        right_knee = points[5]

        shoulder_y = (
            left_shoulder[1] +
            right_shoulder[1]
        ) / 2.0

        hip_y = (
            left_hip[1] +
            right_hip[1]
        ) / 2.0

        knee_y = (
            left_knee[1] +
            right_knee[1]
        ) / 2.0

        height = box_height(person_box)

        torso_length = abs(hip_y - shoulder_y) / height

        hip_to_knee = abs(knee_y - hip_y) / height

        # Sitting generally has a compressed torso
        # and different hip/knee relationship.
        if torso_length < 0.25 and hip_to_knee < 0.30:
            return "sitting"

        return "standing"

    except Exception:
        return None


# ============================================================
# TRACK CLASS
# ============================================================

class Track:

    def __init__(self, track_id, box, frame_number, timestamp):

        self.track_id = track_id

        self.box = box

        self.last_box = box

        self.first_frame = frame_number

        self.last_frame = frame_number

        self.first_time = timestamp

        self.last_time = timestamp

        self.missed_frames = 0

        self.total_frames = 1

        self.positions = deque(
            maxlen=MOVEMENT_HISTORY
        )

        center = box_center(box)

        self.positions.append(center)

        self.activity_history = []

        self.current_activity = None

        self.current_activity_start = timestamp

        self.activities = []

        self.max_confidence = 0.0

    def update(
        self,
        box,
        frame_number,
        timestamp,
        confidence
    ):

        self.last_box = self.box

        self.box = box

        self.last_frame = frame_number

        self.last_time = timestamp

        self.missed_frames = 0

        self.total_frames += 1

        center = box_center(box)

        self.positions.append(center)

        self.max_confidence = max(
            self.max_confidence,
            confidence
        )

    def movement(self):

        if len(self.positions) < 2:
            return 0.0

        total_distance = 0.0

        positions = list(self.positions)

        for i in range(1, len(positions)):

            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]

            distance = math.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )

            total_distance += distance

        normalized = total_distance / (
            box_height(self.box) *
            max(1, len(positions) - 1)
        )

        return normalized

    def update_activity(
        self,
        activity,
        timestamp
    ):

        if activity is None:
            return

        # First activity
        if self.current_activity is None:

            self.current_activity = activity

            self.current_activity_start = timestamp

            return

        # Same activity
        if activity == self.current_activity:

            return

        # Ignore extremely short changes.
        if timestamp - self.current_activity_start < 0.20:

            return

        # Close previous activity
        self.activities.append(
            {
                "activity": self.current_activity,
                "start_time": round(
                    self.current_activity_start,
                    2
                ),
                "end_time": round(
                    timestamp,
                    2
                ),
                "duration_seconds": round(
                    timestamp -
                    self.current_activity_start,
                    2
                )
            }
        )

        # Start new activity
        self.current_activity = activity

        self.current_activity_start = timestamp

    def finalize(self, video_duration):

        if self.current_activity is not None:

            end_time = min(
                video_duration,
                self.last_time
            )

            if end_time > self.current_activity_start:

                self.activities.append(
                    {
                        "activity": self.current_activity,
                        "start_time": round(
                            self.current_activity_start,
                            2
                        ),
                        "end_time": round(
                            end_time,
                            2
                        ),
                        "duration_seconds": round(
                            end_time -
                            self.current_activity_start,
                            2
                        )
                    }
                )

        # Merge consecutive same activities
        merged = []

        for activity in self.activities:

            if not merged:

                merged.append(activity)

                continue

            previous = merged[-1]

            if previous["activity"] == activity["activity"]:

                previous["end_time"] = activity["end_time"]

                previous["duration_seconds"] = round(
                    previous["end_time"] -
                    previous["start_time"],
                    2
                )

            else:

                merged.append(activity)

        self.activities = merged


# ============================================================
# TRACK MANAGER
# ============================================================

class Tracker:

    def __init__(self):

        self.tracks = {}

        self.next_id = 1

    def update(
        self,
        detections,
        frame_number,
        timestamp
    ):

        # detections:
        # [(box, confidence), ...]

        active_ids = list(self.tracks.keys())

        matched_tracks = set()

        matched_detections = set()

        candidates = []

        # ----------------------------------------------------
        # Calculate matching scores
        # ----------------------------------------------------

        for track_id in active_ids:

            track = self.tracks[track_id]

            for detection_index, (
                box,
                confidence
            ) in enumerate(detections):

                iou = calculate_iou(
                    track.box,
                    box
                )

                distance = center_distance(
                    track.box,
                    box
                )

                allowed_distance = (
                    CENTER_DISTANCE_THRESHOLD *
                    box_height(track.box)
                )

                if (
                    iou >= TRACK_IOU_THRESHOLD
                    or distance <= allowed_distance
                ):

                    # Higher is better
                    score = (
                        iou * 2.0
                        -
                        distance /
                        max(
                            box_height(track.box),
                            1
                        )
                        * 0.15
                    )

                    candidates.append(
                        (
                            score,
                            track_id,
                            detection_index
                        )
                    )

        # Best matches first
        candidates.sort(
            reverse=True
        )

        # ----------------------------------------------------
        # Assign detections to tracks
        # ----------------------------------------------------

        for (
            score,
            track_id,
            detection_index
        ) in candidates:

            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            box, confidence = detections[
                detection_index
            ]

            self.tracks[track_id].update(
                box,
                frame_number,
                timestamp,
                confidence
            )

            matched_tracks.add(track_id)

            matched_detections.add(
                detection_index
            )

        # ----------------------------------------------------
        # Create new tracks
        # ----------------------------------------------------

        for detection_index, (
            box,
            confidence
        ) in enumerate(detections):

            if detection_index in matched_detections:
                continue

            track = Track(
                self.next_id,
                box,
                frame_number,
                timestamp
            )

            track.max_confidence = confidence

            self.tracks[
                self.next_id
            ] = track

            self.next_id += 1

        # ----------------------------------------------------
        # Increase missed count
        # ----------------------------------------------------

        for track_id in list(
            self.tracks.keys()
        ):

            if track_id not in matched_tracks:

                self.tracks[
                    track_id
                ].missed_frames += 1

        # ----------------------------------------------------
        # Remove tracks that disappeared too long
        # ----------------------------------------------------

        # IMPORTANT:
        # We do NOT delete the historical person.
        # We only mark the tracking object inactive.
        #
        # The final JSON still contains that person.

        return [
            self.tracks[track_id]
            for track_id in self.tracks
            if self.tracks[
                track_id
            ].missed_frames == 0
        ]


# ============================================================
# LOAD VIDEO
# ============================================================

print()
print("=" * 70)
print("        ALL-PERSON ACTIVITY ANALYZER")
print("=" * 70)
print()

if not os.path.exists(VIDEO_PATH):

    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

print("Loading detection model...")

detector = YOLO(
    DETECTION_MODEL_PATH
)

print("Detection model loaded.")

print()
print("Loading pose model...")

pose_model = YOLO(
    POSE_MODEL_PATH
)

print("Pose model loaded.")

print()
print("Opening video...")

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open video."
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0

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

video_duration = (
    total_frames / fps
)

print()
print("VIDEO INFORMATION")
print("-" * 70)
print(f"Resolution       : {width} x {height}")
print(f"FPS              : {fps:.2f}")
print(f"Total frames     : {total_frames}")
print(
    f"Duration         : {video_duration:.2f} seconds"
)
print()


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    ANNOTATED_VIDEO_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACKER
# ============================================================

tracker = Tracker()


# ============================================================
# STATISTICS
# ============================================================

frame_number = 0

max_people_in_frame = 0

total_person_detections = 0

activity_counts = defaultdict(int)

last_pose_boxes = []

last_pose_keypoints = []


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    timestamp = (
        frame_number - 1
    ) / fps

    # --------------------------------------------------------
    # PERSON DETECTION
    # --------------------------------------------------------

    detection_results = detector.predict(
        source=frame,
        conf=CONFIDENCE,
        iou=IOU,
        imgsz=IMAGE_SIZE,
        classes=[0],
        verbose=False
    )

    result = detection_results[0]

    detections = []

    if result.boxes is not None:

        for box_data in result.boxes:

            confidence = float(
                box_data.conf[0]
            )

            xyxy = box_data.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = map(
                float,
                xyxy
            )

            # Clamp coordinates
            x1 = max(
                0,
                min(width - 1, x1)
            )

            y1 = max(
                0,
                min(height - 1, y1)
            )

            x2 = max(
                0,
                min(width - 1, x2)
            )

            y2 = max(
                0,
                min(height - 1, y2)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                (
                    [x1, y1, x2, y2],
                    confidence
                )
            )

    total_person_detections += len(
        detections
    )

    max_people_in_frame = max(
        max_people_in_frame,
        len(detections)
    )

    # --------------------------------------------------------
    # POSE DETECTION
    # --------------------------------------------------------

    if (
        frame_number % POSE_INTERVAL == 0
    ):

        pose_results = pose_model.predict(
            source=frame,
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMAGE_SIZE,
            classes=[0],
            verbose=False
        )

        pose_result = pose_results[0]

        last_pose_boxes = []
        last_pose_keypoints = []

        if (
            pose_result.boxes is not None
            and
            pose_result.keypoints is not None
        ):

            pose_boxes = (
                pose_result
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            pose_points = (
                pose_result
                .keypoints
                .data
                .cpu()
                .numpy()
            )

            for i in range(
                len(pose_boxes)
            ):

                last_pose_boxes.append(
                    pose_boxes[i].tolist()
                )

                last_pose_keypoints.append(
                    pose_points[i]
                )

    # --------------------------------------------------------
    # UPDATE TRACKER
    # --------------------------------------------------------

    active_tracks = tracker.update(
        detections,
        frame_number,
        timestamp
    )

    # --------------------------------------------------------
    # CLASSIFY EVERY DETECTED PERSON
    # --------------------------------------------------------

    for track in active_tracks:

        pose_keypoints = (
            get_pose_for_person(
                track.box,
                last_pose_boxes,
                last_pose_keypoints
            )
        )

        pose_activity = classify_from_pose(
            track.box,
            pose_keypoints
        )

        movement = track.movement()

        # ----------------------------------------------------
        # Activity decision
        # ----------------------------------------------------

        if movement >= WALKING_MOVEMENT_THRESHOLD:

            activity = "walking"

        elif pose_activity == "sitting":

            activity = "sitting"

        else:

            activity = "standing"

        track.activity_history.append(
            activity
        )

        # ----------------------------------------------------
        # Smooth activity
        # ----------------------------------------------------

        recent = track.activity_history[
            -7:
        ]

        if recent:

            counts = {}

            for item in recent:

                counts[item] = (
                    counts.get(item, 0) + 1
                )

            stable_activity = max(
                counts,
                key=counts.get
            )

        else:

            stable_activity = activity

        track.update_activity(
            stable_activity,
            timestamp
        )

    # --------------------------------------------------------
    # DRAW OUTPUT
    # --------------------------------------------------------

    output_frame = frame.copy()

    current_person_count = len(
        detections
    )

    # Header
    cv2.rectangle(
        output_frame,
        (0, 0),
        (width, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output_frame,
        f"Persons in frame: {current_person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output_frame,
        f"Frame: {frame_number}/{total_frames}  "
        f"Time: {timestamp:.2f}s",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # Draw all active tracks
    for track in active_tracks:

        x1, y1, x2, y2 = map(
            int,
            track.box
        )

        # Get current activity
        if track.activity_history:

            activity = track.activity_history[-1]

        else:

            activity = "unknown"

        # Activity-based drawing colors
        if activity == "walking":

            color = (0, 255, 0)

        elif activity == "sitting":

            color = (255, 0, 0)

        else:

            color = (0, 255, 255)

        cv2.rectangle(
            output_frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f"ID {track.track_id} | "
            f"{activity}"
        )

        cv2.rectangle(
            output_frame,
            (x1, max(0, y1 - 30)),
            (
                min(
                    width - 1,
                    x1 + 220
                ),
                y1
            ),
            color,
            -1
        )

        cv2.putText(
            output_frame,
            label,
            (x1 + 5, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )

    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    if SAVE_ALL_FRAMES:

        frame_path = os.path.join(
            FRAMES_DIR,
            f"frame_{frame_number:06d}.jpg"
        )

        cv2.imwrite(
            frame_path,
            output_frame
        )

    # --------------------------------------------------------
    # SAVE SNAPSHOT
    # --------------------------------------------------------

    if (
        frame_number % SNAPSHOT_INTERVAL == 0
    ):

        snapshot_path = os.path.join(
            SNAPSHOTS_DIR,
            f"snapshot_{frame_number:06d}.jpg"
        )

        cv2.imwrite(
            snapshot_path,
            output_frame
        )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    writer.write(
        output_frame
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        frame_number % 25 == 0
        or
        frame_number == total_frames
    ):

        print(
            f"Processed "
            f"{frame_number}/{total_frames} "
            f"| persons in frame: "
            f"{current_person_count} "
            f"| tracked IDs: "
            f"{len(tracker.tracks)}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()

writer.release()


# ============================================================
# FINALIZE TRACKS
# ============================================================

for track in tracker.tracks.values():

    track.finalize(
        video_duration
    )


# ============================================================
# REMOVE VERY SHORT TRACKS
# ============================================================

valid_tracks = []

for track in tracker.tracks.values():

    if (
        track.total_frames
        >= MIN_TRACK_FRAMES
    ):

        valid_tracks.append(
            track
        )


# ============================================================
# SORT TRACKS
# ============================================================

valid_tracks.sort(
    key=lambda x: x.track_id
)


# ============================================================
# CALCULATE ACTIVITY SUMMARY
# ============================================================

walking_count = 0
standing_count = 0
sitting_count = 0


persons_json = []


for track in valid_tracks:

    activities = track.activities

    # If no activity segment was created,
    # create one using the last known state.
    if not activities:

        if track.activity_history:

            activity = max(
                set(
                    track.activity_history
                ),
                key=track.activity_history.count
            )

        else:

            activity = "standing"

        activities = [
            {
                "activity": activity,
                "start_time": round(
                    track.first_time,
                    2
                ),
                "end_time": round(
                    track.last_time,
                    2
                ),
                "duration_seconds": round(
                    max(
                        0,
                        track.last_time -
                        track.first_time
                    ),
                    2
                )
            }
        ]

    # --------------------------------------------------------
    # Determine person's dominant activity
    # --------------------------------------------------------

    activity_duration = defaultdict(float)

    for item in activities:

        activity_duration[
            item["activity"]
        ] += item[
            "duration_seconds"
        ]

    if activity_duration:

        dominant_activity = max(
            activity_duration,
            key=activity_duration.get
        )

    else:

        dominant_activity = "standing"

    if dominant_activity == "walking":

        walking_count += 1

    elif dominant_activity == "sitting":

        sitting_count += 1

    else:

        standing_count += 1

    # --------------------------------------------------------
    # Person JSON
    # --------------------------------------------------------

    persons_json.append(
        {
            "person_id": track.track_id,
            "activities": activities
        }
    )


# ============================================================
# FINAL JSON
# ============================================================

final_json = {

    "video_name":
        os.path.basename(
            VIDEO_PATH
        ),

    "video_duration_seconds":
        round(
            video_duration,
            2
        ),

    "summary":
        {
            "total_unique_persons":
                len(valid_tracks),

            "walking":
                walking_count,

            "standing":
                standing_count,

            "sitting":
                sitting_count
        },

    "persons":
        persons_json
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
        final_json,
        file,
        indent=4
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("          ACTIVITY ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    f"Video duration            : "
    f"{video_duration:.2f} seconds"
)

print(
    f"Frames processed          : "
    f"{frame_number}"
)

print(
    f"Maximum persons/frame     : "
    f"{max_people_in_frame}"
)

print(
    f"Total person detections   : "
    f"{total_person_detections}"
)

print(
    f"Unique persons             : "
    f"{len(valid_tracks)}"
)

print()
print("ACTIVITY SUMMARY")
print("-" * 70)

print(
    f"Walking                   : "
    f"{walking_count}"
)

print(
    f"Standing                  : "
    f"{standing_count}"
)

print(
    f"Sitting                   : "
    f"{sitting_count}"
)

print()
print("OUTPUT FILES")
print("-" * 70)

print(
    f"JSON                      : "
    f"{JSON_OUTPUT_PATH}"
)

print(
    f"Annotated video            : "
    f"{ANNOTATED_VIDEO_PATH}"
)

print(
    f"All frames                : "
    f"{FRAMES_DIR}"
)

print(
    f"Snapshots                 : "
    f"{SNAPSHOTS_DIR}"
)

print()
print("=" * 70)