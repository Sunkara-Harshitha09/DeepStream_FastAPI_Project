import os
import json
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

JSON_FOLDER = os.path.join(
    BASE_DIR,
    "json_output"
)


# ============================================================
# FASTAPI URL
# ============================================================

API_URL = (
    "http://127.0.0.1:8000/detection"
)


# ============================================================
# GET AVAILABLE JSON FILES
# ============================================================

def get_json_files():

    if not os.path.exists(JSON_FOLDER):

        return []

    json_files = []

    for file_name in os.listdir(
        JSON_FOLDER
    ):

        if file_name.lower().endswith(
            ".json"
        ):

            json_files.append(
                file_name
            )

    json_files.sort()

    return json_files


# ============================================================
# SELECT JSON FILE
# ============================================================

def select_json_file():

    json_files = get_json_files()

    if not json_files:

        print(
            "\nNo JSON files found."
        )

        print(
            "First generate detection JSON "
            "using detector.py."
        )

        return None

    print("\n" + "=" * 60)
    print("              AVAILABLE JSON FILES")
    print("=" * 60)

    for index, file_name in enumerate(
        json_files,
        start=1
    ):

        print(
            f"{index}. {file_name}"
        )

    print(
        "0. Exit"
    )

    while True:

        choice = input(
            "\nSelect JSON file: "
        ).strip()

        if choice == "0":

            return None

        if not choice.isdigit():

            print(
                "Please enter a valid number."
            )

            continue

        choice = int(choice)

        if 1 <= choice <= len(json_files):

            selected_file = os.path.join(
                JSON_FOLDER,
                json_files[choice - 1]
            )

            return selected_file

        print(
            "Invalid selection."
        )


# ============================================================
# SEND JSON TO FASTAPI
# ============================================================

def send_json_to_fastapi(
    json_file
):

    if json_file is None:

        return

    if not os.path.exists(
        json_file
    ):

        print(
            "\nJSON file does not exist."
        )

        return

    try:

        # ----------------------------------------------------
        # READ JSON FILE
        # ----------------------------------------------------

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as file:

            json_data = json.load(
                file
            )

        print("\n" + "=" * 60)
        print("           SENDING JSON TO FASTAPI")
        print("=" * 60)

        print(
            f"\nJSON file:"
        )

        print(
            json_file
        )

        print(
            f"\nVideo:"
        )

        print(
            json_data.get(
                "video_name",
                "Unknown"
            )
        )

        print(
            f"\nTotal frames:"
        )

        print(
            json_data.get(
                "total_frames",
                0
            )
        )

        # ----------------------------------------------------
        # COUNT OBJECTS
        # ----------------------------------------------------

        total_objects = 0

        for frame in json_data.get(
            "detections",
            []
        ):

            total_objects += len(
                frame.get(
                    "objects",
                    []
                )
            )

        print(
            f"Total detected objects:"
        )

        print(
            total_objects
        )

        print(
            "\nSending request..."
        )

        # ----------------------------------------------------
        # SEND POST REQUEST
        # ----------------------------------------------------

        response = requests.post(

            API_URL,

            json=json_data,

            timeout=300
        )

        # ----------------------------------------------------
        # DISPLAY RESPONSE
        # ----------------------------------------------------

        print(
            "\nHTTP Status:"
        )

        print(
            response.status_code
        )

        print(
            "\nFastAPI Response:"
        )

        try:

            response_data = (
                response.json()
            )

            print(
                json.dumps(
                    response_data,
                    indent=4
                )
            )

        except ValueError:

            print(
                response.text
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.ok:

            print(
                "\n" + "=" * 60
            )

            print(
                "       JSON SENT SUCCESSFULLY"
            )

            print(
                "=" * 60
            )

        else:

            print(
                "\n" + "=" * 60
            )

            print(
                "       FASTAPI REQUEST FAILED"
            )

            print(
                "=" * 60
            )

    except requests.exceptions.ConnectionError:

        print(
            "\nCould not connect to FastAPI."
        )

        print(
            "Make sure Uvicorn is running:"
        )

        print(
            "uvicorn fastapi_app.main:app --reload"
        )

    except requests.exceptions.Timeout:

        print(
            "\nFastAPI request timed out."
        )

    except json.JSONDecodeError:

        print(
            "\nThe JSON file is invalid."
        )

    except Exception as error:

        print(
            "\nUnexpected error:"
        )

        print(
            error
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)

    print(
        "        YOLO JSON → FASTAPI CLIENT"
    )

    print("=" * 60)

    json_file = select_json_file()

    if json_file is None:

        print(
            "\nNo JSON selected."
        )

        return

    print(
        f"\nSelected:"
    )

    print(
        json_file
    )

    send_json_to_fastapi(
        json_file
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()