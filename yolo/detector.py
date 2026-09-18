import os
import cv2
import json
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VIDEO_FOLDER = os.path.join(
    BASE_DIR,
    "videos"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "json_output"
)

DETECTION_FOLDER = os.path.join(
    BASE_DIR,
    "runs",
    "detect"
)


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_NAME = "yolo11n.pt"

print("\nLoading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded successfully.")


# ============================================================
# DISPLAY TITLE
# ============================================================

def display_title():

    print("\n" + "=" * 70)
    print("                 YOLO VIDEO DETECTION SYSTEM")
    print("=" * 70)


# ============================================================
# GET AVAILABLE VIDEOS
# ============================================================

def get_videos():

    if not os.path.exists(VIDEO_FOLDER):
        return []

    supported_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    )

    videos = []

    for file_name in os.listdir(VIDEO_FOLDER):

        if file_name.lower().endswith(
            supported_extensions
        ):

            videos.append(file_name)

    videos.sort()

    return videos


# ============================================================
# SELECT VIDEO
# ============================================================

def select_video():

    videos = get_videos()

    if not videos:

        print("\nNo videos found inside the videos folder.")

        return None

    print("\nAvailable Videos")
    print("-" * 45)

    for index, video in enumerate(
        videos,
        start=1
    ):

        print(f"{index}. {video}")

    print("0. Back")

    while True:

        choice = input(
            "\nSelect video: "
        ).strip()

        if choice == "0":

            return None

        if not choice.isdigit():

            print(
                "Please enter a valid number."
            )

            continue

        choice = int(choice)

        if 1 <= choice <= len(videos):

            selected_video = os.path.join(
                VIDEO_FOLDER,
                videos[choice - 1]
            )

            print(
                f"\nSelected video: "
                f"{videos[choice - 1]}"
            )

            return selected_video

        print(
            "Invalid selection."
        )


# ============================================================
# VIEW VIDEO
# ============================================================

def view_video(
    video_path,
    window_title="Video"
):

    if video_path is None:

        print(
            "\nPlease select a video first."
        )

        return

    if not os.path.exists(video_path):

        print(
            "\nVideo file does not exist."
        )

        return

    print("\nOpening video...")
    print("Press Q to close the video.")

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "\nUnable to open video."
        )

        return

    # --------------------------------------------------------
    # GET ORIGINAL VIDEO DIMENSIONS
    # --------------------------------------------------------

    video_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    video_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # --------------------------------------------------------
    # CREATE RESIZABLE WINDOW
    # --------------------------------------------------------

    cv2.namedWindow(
        window_title,
        cv2.WINDOW_NORMAL
    )

    # --------------------------------------------------------
    # MAXIMUM DISPLAY SIZE
    # --------------------------------------------------------

    max_width = 1200
    max_height = 700

    # Prevent division by zero
    if video_width <= 0:
        video_width = 640

    if video_height <= 0:
        video_height = 480

    # --------------------------------------------------------
    # CALCULATE SCALE
    # --------------------------------------------------------

    scale_width = (
        max_width / video_width
    )

    scale_height = (
        max_height / video_height
    )

    scale = min(
        scale_width,
        scale_height,
        1
    )

    display_width = int(
        video_width * scale
    )

    display_height = int(
        video_height * scale
    )

    # --------------------------------------------------------
    # SET WINDOW SIZE
    # --------------------------------------------------------

    cv2.resizeWindow(
        window_title,
        display_width,
        display_height
    )

    # --------------------------------------------------------
    # PLAY VIDEO
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        # Resize only when necessary
        if scale < 1:

            frame = cv2.resize(
                frame,
                (
                    display_width,
                    display_height
                ),
                interpolation=cv2.INTER_AREA
            )

        cv2.imshow(
            window_title,
            frame
        )

        # ----------------------------------------------------
        # Q = EXIT VIDEO
        # ----------------------------------------------------

        if (
            cv2.waitKey(25) & 0xFF
            == ord("q")
        ):

            break

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()

    cv2.destroyWindow(
        window_title
    )

    cv2.waitKey(1)


# ============================================================
# RUN YOLO DETECTION
# ============================================================

def run_yolo_detection(video_path):

    if video_path is None:

        print(
            "\nPlease select a video first."
        )

        return None

    if not os.path.exists(video_path):

        print(
            "\nVideo file does not exist."
        )

        return None

    video_name = os.path.basename(
        video_path
    )

    print("\n" + "=" * 70)

    print(
        "                 RUNNING YOLO DETECTION"
    )

    print("=" * 70)

    print(
        f"\nInput video : {video_name}"
    )

    print(
        "Model       : YOLO11n"
    )

    print(
        "Device      : CPU"
    )

    print(
        "\nProcessing video..."
    )

    print(
        "Please wait...\n"
    )

    try:

        results = model.predict(
            source=video_path,
            save=True,
            verbose=True
        )

        print("\n" + "=" * 70)

        print(
            "              YOLO DETECTION COMPLETED"
        )

        print("=" * 70)

        output_video = (
            get_latest_detection_video()
        )

        if output_video:

            print(
                "\nOutput video:"
            )

            print(
                output_video
            )

        return results

    except Exception as error:

        print(
            "\nYOLO detection failed."
        )

        print(
            f"Error: {error}"
        )

        return None


# ============================================================
# FIND LATEST DETECTION VIDEO
# ============================================================

def get_latest_detection_video():

    if not os.path.exists(
        DETECTION_FOLDER
    ):

        return None

    prediction_folders = []

    for folder in os.listdir(
        DETECTION_FOLDER
    ):

        folder_path = os.path.join(
            DETECTION_FOLDER,
            folder
        )

        if os.path.isdir(
            folder_path
        ):

            prediction_folders.append(
                folder_path
            )

    if not prediction_folders:

        return None

    latest_folder = max(
        prediction_folders,
        key=os.path.getmtime
    )

    supported_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    )

    video_files = []

    for file_name in os.listdir(
        latest_folder
    ):

        if file_name.lower().endswith(
            supported_extensions
        ):

            video_files.append(
                os.path.join(
                    latest_folder,
                    file_name
                )
            )

    if not video_files:

        return None

    return max(
        video_files,
        key=os.path.getmtime
    )


# ============================================================
# VIEW DETECTION OUTPUT
# ============================================================

def view_detection_output():

    output_video = (
        get_latest_detection_video()
    )

    if output_video is None:

        print(
            "\nNo detection output found."
        )

        print(
            "Run YOLO Detection first."
        )

        return

    print(
        "\nDetection Output:"
    )

    print(
        output_video
    )

    view_video(
        output_video,
        "YOLO Detection Output"
    )


# ============================================================
# GENERATE DETECTION JSON
# ============================================================

def generate_detection_json(video_path):

    if video_path is None:

        print(
            "\nPlease select a video first."
        )

        return None

    if not os.path.exists(video_path):

        print(
            "\nVideo file does not exist."
        )

        return None

    video_name = os.path.basename(
        video_path
    )

    print("\n" + "=" * 70)

    print(
        "              GENERATING DETECTION JSON"
    )

    print("=" * 70)

    print(
        f"\nVideo: {video_name}"
    )

    print(
        "\nRunning YOLO frame-by-frame..."
    )

    print(
        "Please wait...\n"
    )

    try:

        # ----------------------------------------------------
        # OPEN VIDEO
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                "Unable to open video."
            )

            return None

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
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

        # ----------------------------------------------------
        # JSON STRUCTURE
        # ----------------------------------------------------

        detection_data = {

            "video_name":
                video_name,

            "video_path":
                video_path,

            "total_frames":
                total_frames,

            "fps":
                fps,

            "width":
                width,

            "height":
                height,

            "generated_at":
                datetime.now().isoformat(),

            "detections":
                []
        }

        # ----------------------------------------------------
        # PROCESS EVERY FRAME
        # ----------------------------------------------------

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            frame_number += 1

            results = model.predict(
                source=frame,
                verbose=False
            )

            frame_detections = []

            for result in results:

                boxes = result.boxes

                if boxes is None:

                    continue

                for box in boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    class_name = (
                        model.names[
                            class_id
                        ]
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    coordinates = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    x1 = float(
                        coordinates[0]
                    )

                    y1 = float(
                        coordinates[1]
                    )

                    x2 = float(
                        coordinates[2]
                    )

                    y2 = float(
                        coordinates[3]
                    )

                    detection = {

                        "class_id":
                            class_id,

                        "class_name":
                            class_name,

                        "confidence":
                            round(
                                confidence,
                                4
                            ),

                        "bounding_box": {

                            "x1":
                                round(
                                    x1,
                                    2
                                ),

                            "y1":
                                round(
                                    y1,
                                    2
                                ),

                            "x2":
                                round(
                                    x2,
                                    2
                                ),

                            "y2":
                                round(
                                    y2,
                                    2
                                )
                        }
                    }

                    frame_detections.append(
                        detection
                    )

            # ------------------------------------------------
            # SAVE FRAME DETECTIONS
            # ------------------------------------------------

            detection_data[
                "detections"
            ].append(

                {

                    "frame_number":
                        frame_number,

                    "timestamp_seconds":
                        round(
                            frame_number / fps,
                            3
                        ) if fps > 0
                        else 0,

                    "objects":
                        frame_detections
                }
            )

            # ------------------------------------------------
            # PROGRESS DISPLAY
            # ------------------------------------------------

            if (
                frame_number % 25 == 0
                or frame_number == total_frames
            ):

                percentage = (

                    frame_number
                    / total_frames
                    * 100

                ) if total_frames > 0 else 0

                print(
                    f"Processed "
                    f"{frame_number}/"
                    f"{total_frames} "
                    f"frames "
                    f"({percentage:.1f}%)"
                )

        cap.release()

        # ----------------------------------------------------
        # JSON FILE NAME
        # ----------------------------------------------------

        base_name = os.path.splitext(
            video_name
        )[0]

        json_file = os.path.join(
            OUTPUT_FOLDER,
            f"{base_name}.json"
        )

        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        with open(
            json_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                detection_data,
                file,
                indent=4
            )

        print("\n" + "=" * 70)

        print(
            "              JSON GENERATION COMPLETED"
        )

        print("=" * 70)

        print(
            "\nJSON saved to:"
        )

        print(
            json_file
        )

        return json_file

    except Exception as error:

        print(
            "\nJSON generation failed."
        )

        print(
            f"Error: {error}"
        )

        return None


# ============================================================
# VIEW DETECTION JSON
# ============================================================

def view_detection_json(video_path):

    if video_path is None:

        print(
            "\nPlease select a video first."
        )

        return

    video_name = os.path.basename(
        video_path
    )

    base_name = os.path.splitext(
        video_name
    )[0]

    json_file = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}.json"
    )

    if not os.path.exists(
        json_file
    ):

        print(
            "\nJSON file not found."
        )

        print(
            "Generate the JSON first "
            "using option 5."
        )

        return

    try:

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print("\n" + "=" * 70)

        print(
            "                  DETECTION JSON"
        )

        print("=" * 70)

        print(
            f"\nVideo        : "
            f"{data['video_name']}"
        )

        print(
            f"Total Frames : "
            f"{data['total_frames']}"
        )

        print(
            f"FPS          : "
            f"{data['fps']:.2f}"
        )

        print(
            f"Resolution   : "
            f"{data['width']} x "
            f"{data['height']}"
        )

        print(
            "\nJSON File:"
        )

        print(
            json_file
        )

        # ----------------------------------------------------
        # COUNT OBJECTS
        # ----------------------------------------------------

        total_objects = 0

        class_counts = {}

        for frame_data in data[
            "detections"
        ]:

            for obj in frame_data[
                "objects"
            ]:

                total_objects += 1

                class_name = obj[
                    "class_name"
                ]

                class_counts[
                    class_name
                ] = class_counts.get(
                    class_name,
                    0
                ) + 1

        print(
            "\nDetection Summary"
        )

        print(
            "-" * 70
        )

        print(
            f"Total detected objects: "
            f"{total_objects}"
        )

        print(
            "\nObject Counts:"
        )

        for class_name, count in sorted(
            class_counts.items()
        ):

            print(
                f"  {class_name}: {count}"
            )

        # ----------------------------------------------------
        # DISPLAY COMPLETE JSON
        # ----------------------------------------------------

        show_full = input(
            "\nDisplay complete JSON? (y/n): "
        ).strip().lower()

        if show_full == "y":

            print(
                "\n" + "=" * 70
            )

            print(
                json.dumps(
                    data,
                    indent=4
                )
            )

            print(
                "=" * 70
            )

    except Exception as error:

        print(
            "\nUnable to read JSON."
        )

        print(
            f"Error: {error}"
        )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    selected_video = None

    while True:

        display_title()

        if selected_video:

            print(
                "Selected Video : "
                + os.path.basename(
                    selected_video
                )
            )

        else:

            print(
                "Selected Video : None"
            )

        print(
            "\n1. Select Video"
        )

        print(
            "2. View Original Video"
        )

        print(
            "3. Run YOLO Detection"
        )

        print(
            "4. View Detection Output"
        )

        print(
            "5. Generate Detection JSON"
        )

        print(
            "6. View Detection JSON"
        )

        print(
            "7. Exit"
        )

        print(
            "-" * 70
        )

        choice = input(
            "Enter your choice: "
        ).strip()

        # ====================================================
        # OPTION 1
        # ====================================================

        if choice == "1":

            selected_video = select_video()

        # ====================================================
        # OPTION 2
        # ====================================================

        elif choice == "2":

            view_video(
                selected_video,
                "Original Video"
            )

        # ====================================================
        # OPTION 3
        # ====================================================

        elif choice == "3":

            run_yolo_detection(
                selected_video
            )

        # ====================================================
        # OPTION 4
        # ====================================================

        elif choice == "4":

            view_detection_output()

        # ====================================================
        # OPTION 5
        # ====================================================

        elif choice == "5":

            generate_detection_json(
                selected_video
            )

        # ====================================================
        # OPTION 6
        # ====================================================

        elif choice == "6":

            view_detection_json(
                selected_video
            )

        # ====================================================
        # OPTION 7
        # ====================================================

        elif choice == "7":

            print(
                "\nExiting YOLO Video Detection System..."
            )

            print(
                "Thank you!"
            )

            break

        # ====================================================
        # INVALID OPTION
        # ====================================================

        else:

            print(
                "\nInvalid choice."
            )

        input(
            "\nPress ENTER to continue..."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()