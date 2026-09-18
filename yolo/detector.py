import os
import json
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEOS_DIR = os.path.join(PROJECT_ROOT, "videos")
JSON_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "json_output")

MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo11n.pt")

FASTAPI_URL = "http://127.0.0.1:8000/detection"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# GLOBAL SELECTED VIDEO
# ============================================================

selected_video = None


# ============================================================
# GET ALL VIDEOS
# ============================================================

def get_video_files():

    supported_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    )

    videos = []

    for filename in os.listdir(VIDEOS_DIR):

        if filename.lower().endswith(supported_extensions):

            videos.append(filename)

    videos.sort()

    return videos


# ============================================================
# SELECT VIDEO
# ============================================================

def select_video():

    global selected_video

    videos = get_video_files()

    if not videos:

        print("\nNo videos found in videos folder.")

        return

    print("\n" + "=" * 70)
    print("                         AVAILABLE VIDEOS")
    print("=" * 70)

    for index, video in enumerate(videos, start=1):

        print(f"{index}. {video}")

    print("=" * 70)

    while True:

        try:

            choice = int(
                input("Enter video number: ")
            )

            if 1 <= choice <= len(videos):

                selected_video = videos[choice - 1]

                print(
                    f"\nSelected Video : {selected_video}"
                )

                break

            else:

                print("Invalid choice.")

        except ValueError:

            print("Please enter a number.")


# ============================================================
# GET VIDEO PATH
# ============================================================

def get_selected_video_path():

    if selected_video is None:

        print("\nPlease select a video first.")

        return None

    return os.path.join(
        VIDEOS_DIR,
        selected_video
    )


# ============================================================
# VIEW VIDEO
# ============================================================

def view_video(video_path, window_title):

    if not os.path.exists(video_path):

        print("\nVideo file not found.")

        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("\nUnable to open video.")

        return

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    max_width = 1200
    max_height = 700

    scale = min(
        max_width / width,
        max_height / height,
        1
    )

    display_width = int(width * scale)
    display_height = int(height * scale)

    cv2.namedWindow(
        window_title,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_title,
        display_width,
        display_height
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        if scale < 1:

            frame = cv2.resize(
                frame,
                (
                    display_width,
                    display_height
                )
            )

        cv2.imshow(
            window_title,
            frame
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# VIEW ORIGINAL VIDEO
# ============================================================

def view_original_video():

    video_path = get_selected_video_path()

    if video_path is None:

        return

    print(
        f"\nOpening original video: {selected_video}"
    )

    print("Press Q to close the video.")

    view_video(
        video_path,
        "Original Video"
    )


# ============================================================
# RUN YOLO DETECTION
# ============================================================

def run_yolo_detection(video_path):

    if not os.path.exists(video_path):

        print(
            f"\nVideo not found: {video_path}"
        )

        return None

    print("\n" + "=" * 70)
    print("                    YOLO DETECTION")
    print("=" * 70)

    print(
        f"\nProcessing: {os.path.basename(video_path)}"
    )

    start_time = time.time()

    try:

        results = model.predict(
            source=video_path,
            save=True,
            verbose=True
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("                 YOLO DETECTION COMPLETE")
        print("=" * 70)

        print(
            f"\nProcessing Time: {elapsed:.2f} seconds"
        )

        print(
            "\nYOLO detection output has been generated."
        )

        return results

    except Exception as error:

        print(
            f"\nYOLO detection failed:\n{error}"
        )

        return None


# ============================================================
# FIND LATEST YOLO OUTPUT
# ============================================================

def find_latest_detection_output(video_filename):

    runs_dir = os.path.join(
        PROJECT_ROOT,
        "runs",
        "detect"
    )

    if not os.path.exists(runs_dir):

        return None

    folders = []

    for folder in os.listdir(runs_dir):

        folder_path = os.path.join(
            runs_dir,
            folder
        )

        if os.path.isdir(folder_path):

            folders.append(folder_path)

    if not folders:

        return None

    folders.sort(
        key=os.path.getmtime,
        reverse=True
    )

    video_base_name = os.path.splitext(
        video_filename
    )[0]

    for folder in folders:

        possible_files = [
            f"{video_base_name}.avi",
            f"{video_base_name}.mp4"
        ]

        for filename in possible_files:

            output_path = os.path.join(
                folder,
                filename
            )

            if os.path.exists(output_path):

                return output_path

    return None


# ============================================================
# VIEW DETECTION OUTPUT
# ============================================================

def view_detection_output():

    if selected_video is None:

        print("\nPlease select a video first.")

        return

    output_path = find_latest_detection_output(
        selected_video
    )

    if output_path is None:

        print(
            "\nDetection output not found."
        )

        print(
            "Run YOLO Detection first."
        )

        return

    print(
        f"\nDetection Output:\n{output_path}"
    )

    print(
        "\nPress Q to close the video."
    )

    view_video(
        output_path,
        "YOLO Detection Output"
    )


# ============================================================
# GENERATE DETECTION JSON
# ============================================================

def generate_detection_json(video_path):

    if not os.path.exists(video_path):

        print(
            f"\nVideo not found: {video_path}"
        )

        return None

    video_filename = os.path.basename(
        video_path
    )

    print("\n" + "=" * 70)
    print("                  GENERATING DETECTION JSON")
    print("=" * 70)

    print(
        f"\nVideo: {video_filename}"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            "\nUnable to open video."
        )

        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:

        fps = 1.0

    detections = []

    frame_number = 0

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        frame_number += 1

        timestamp_seconds = (
            (frame_number - 1) / fps
        )

        results = model.predict(
            source=frame,
            verbose=False
        )

        objects = []

        for result in results:

            if result.boxes is None:

                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                class_name = result.names[
                    class_id
                ]

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = box.xyxy[
                    0
                ].tolist()

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

                objects.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(
                            confidence,
                            4
                        ),
                        "bounding_box": {
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

        detections.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": round(
                    timestamp_seconds,
                    3
                ),
                "objects": objects
            }
        )

        if frame_number % 50 == 0:

            print(
                f"Processed {frame_number}/{total_frames} frames..."
            )

    cap.release()

    elapsed = time.time() - start_time

    detection_json = {

        "video_name": video_filename,

        "video_path": video_path,

        "total_frames": total_frames,

        "fps": round(
            fps,
            3
        ),

        "width": width,

        "height": height,

        "generated_at":
            datetime.now().isoformat(),

        "detections": detections
    }

    base_name = os.path.splitext(
        video_filename
    )[0]

    json_path = os.path.join(
        JSON_OUTPUT_DIR,
        f"{base_name}.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            detection_json,
            file,
            indent=4
        )

    total_objects = sum(
        len(frame["objects"])
        for frame in detections
    )

    print("\n" + "=" * 70)
    print("                 JSON GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"\nVideo          : {video_filename}"
    )

    print(
        f"Total Frames   : {total_frames}"
    )

    print(
        f"Total Objects  : {total_objects}"
    )

    print(
        f"Processing Time: {elapsed:.2f} seconds"
    )

    print(
        f"JSON File      : {json_path}"
    )

    print("=" * 70)

    return json_path


# ============================================================
# VIEW JSON
# ============================================================

def view_detection_json():

    if selected_video is None:

        print("\nPlease select a video first.")

        return

    base_name = os.path.splitext(
        selected_video
    )[0]

    json_path = os.path.join(
        JSON_OUTPUT_DIR,
        f"{base_name}.json"
    )

    if not os.path.exists(json_path):

        print(
            "\nDetection JSON not found."
        )

        print(
            "Generate the JSON first."
        )

        return

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    total_objects = sum(
        len(frame["objects"])
        for frame in data["detections"]
    )

    print("\n" + "=" * 70)
    print("                    DETECTION JSON")
    print("=" * 70)

    print(
        f"\nVideo          : {data['video_name']}"
    )

    print(
        f"Total Frames   : {data['total_frames']}"
    )

    print(
        f"FPS            : {data['fps']}"
    )

    print(
        f"Resolution     : {data['width']} x {data['height']}"
    )

    print(
        f"Total Objects  : {total_objects}"
    )

    print(
        f"\nJSON File:\n{json_path}"
    )

    print("=" * 70)

    input(
        "\nPress ENTER to continue..."
    )


# ============================================================
# SEND JSON TO FASTAPI
# ============================================================

def send_json_to_fastapi(json_path):

    try:

        import requests

    except ImportError:

        print(
            "\nThe requests package is not installed."
        )

        print(
            "Run: pip install requests"
        )

        return False

    if not os.path.exists(json_path):

        print(
            f"\nJSON file not found: {json_path}"
        )

        return False

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print("\n" + "=" * 70)
        print("                 SENDING JSON TO FASTAPI")
        print("=" * 70)

        print(
            f"\nVideo: {data.get('video_name')}"
        )

        print(
            f"Sending to: {FASTAPI_URL}"
        )

        response = requests.post(
            FASTAPI_URL,
            json=data,
            timeout=300
        )

        print(
            f"\nHTTP Status: {response.status_code}"
        )

        try:

            response_data = response.json()

            print("\nFastAPI Response:")

            print(
                json.dumps(
                    response_data,
                    indent=4
                )
            )

        except ValueError:

            print(
                "\nFastAPI Response:"
            )

            print(
                response.text
            )

        if response.status_code == 200:

            print(
                "\nJSON successfully processed by FastAPI."
            )

            return True

        print(
            "\nFastAPI rejected the JSON."
        )

        return False

    except requests.exceptions.ConnectionError:

        print(
            "\nUnable to connect to FastAPI."
        )

        print(
            "Make sure FastAPI is running."
        )

        return False

    except requests.exceptions.Timeout:

        print(
            "\nFastAPI request timed out."
        )

        return False

    except Exception as error:

        print(
            f"\nError while sending JSON:\n{error}"
        )

        return False


# ============================================================
# PROCESS ONE VIDEO COMPLETELY
# ============================================================

def process_single_video(video_filename):

    video_path = os.path.join(
        VIDEOS_DIR,
        video_filename
    )

    print("\n\n" + "=" * 80)
    print(
        f"             PROCESSING VIDEO: {video_filename}"
    )
    print("=" * 80)

    # Step 1: YOLO detection
    detection_result = run_yolo_detection(
        video_path
    )

    if detection_result is None:

        print(
            f"\nSkipping {video_filename} because YOLO failed."
        )

        return False

    # Step 2: Generate JSON
    json_path = generate_detection_json(
        video_path
    )

    if json_path is None:

        print(
            f"\nSkipping {video_filename} because JSON generation failed."
        )

        return False

    # Step 3: Send JSON to FastAPI
    success = send_json_to_fastapi(
        json_path
    )

    if success:

        print("\n" + "=" * 80)
        print(
            f"        COMPLETED: {video_filename}"
        )
        print("=" * 80)

        return True

    print("\n" + "=" * 80)
    print(
        f"        FAILED TO SEND: {video_filename}"
    )
    print("=" * 80)

    return False


# ============================================================
# PROCESS ALL VIDEOS
# ============================================================

def process_all_videos():

    videos = get_video_files()

    if not videos:

        print(
            "\nNo videos found in videos folder."
        )

        return

    print("\n" + "=" * 80)
    print("                 PROCESSING ALL VIDEOS")
    print("=" * 80)

    print(
        f"\nTotal Videos Found: {len(videos)}"
    )

    print("\nVideos:")

    for index, video in enumerate(
        videos,
        start=1
    ):

        print(
            f"{index}. {video}"
        )

    print("\n" + "=" * 80)

    confirmation = input(
        "\nStart processing all videos? (y/n): "
    ).strip().lower()

    if confirmation != "y":

        print(
            "\nProcessing cancelled."
        )

        return

    overall_start = time.time()

    successful = []
    failed = []

    for index, video in enumerate(
        videos,
        start=1
    ):

        print(
            f"\n\nVIDEO {index}/{len(videos)}"
        )

        success = process_single_video(
            video
        )

        if success:

            successful.append(video)

        else:

            failed.append(video)

    overall_elapsed = (
        time.time() - overall_start
    )

    print("\n\n" + "=" * 80)
    print("                 ALL VIDEO PROCESSING COMPLETE")
    print("=" * 80)

    print(
        f"\nTotal Videos : {len(videos)}"
    )

    print(
        f"Successful   : {len(successful)}"
    )

    print(
        f"Failed       : {len(failed)}"
    )

    print(
        f"Total Time   : {overall_elapsed:.2f} seconds"
    )

    if successful:

        print("\nSuccessful Videos:")

        for video in successful:

            print(
                f"  ✓ {video}"
            )

    if failed:

        print("\nFailed Videos:")

        for video in failed:

            print(
                f"  ✗ {video}"
            )

    print("=" * 80)


# ============================================================
# MAIN MENU
# ============================================================

def main():

    global selected_video

    while True:

        print("\n" + "=" * 70)
        print("                 YOLO VIDEO DETECTION SYSTEM")
        print("=" * 70)

        if selected_video:

            print(
                f"Selected Video : {selected_video}"
            )

        else:

            print(
                "Selected Video : None"
            )

        print("\n1. Select Video")
        print("2. View Original Video")
        print("3. Run YOLO Detection")
        print("4. View Detection Output")
        print("5. Generate Detection JSON")
        print("6. View Detection JSON")
        print("7. Process ALL Videos")
        print("8. Exit")

        print("=" * 70)

        choice = input(
            "Enter your choice: "
        ).strip()

        if choice == "1":

            select_video()

        elif choice == "2":

            view_original_video()

        elif choice == "3":

            video_path = get_selected_video_path()

            if video_path:

                run_yolo_detection(
                    video_path
                )

        elif choice == "4":

            view_detection_output()

        elif choice == "5":

            video_path = get_selected_video_path()

            if video_path:

                generate_detection_json(
                    video_path
                )

        elif choice == "6":

            view_detection_json()

        elif choice == "7":

            process_all_videos()

        elif choice == "8":

            print("\nThank you!")

            break

        else:

            print(
                "\nInvalid choice. Please select 1-8."
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()