# DeepStream FastAPI Project

An end-to-end AI video detection pipeline that processes videos using YOLO object detection, generates structured JSON detection results, sends the results to FastAPI, validates the received data, and stores the validated detection events in PostgreSQL.

The project is designed with a reusable architecture so that YOLO-based inference can later be replaced with NVIDIA DeepStream when an NVIDIA GPU is available.

---

## 1. Project Overview

The main objective of this project is to build a complete AI video-processing pipeline:

Videos → AI Detection → JSON → FastAPI → Validation → PostgreSQL

Currently, YOLO is used for object detection because the development machine does not have an NVIDIA GPU.

The planned production pipeline is:

5 Videos
↓
NVIDIA DeepStream
↓
AI Object Detection
↓
Detection JSON
↓
FastAPI
↓
JSON Validation
↓
PostgreSQL
↓
detection_events Table

The current development pipeline is:

5 Videos
↓
YOLO
↓
Detection JSON
↓
FastAPI
↓
JSON Validation
↓
PostgreSQL
↓
detection_events Table

---

## 2. Objectives

The project implements the following requirements:

1. Take five input videos.
2. Process the videos using an AI object detection model.
3. Generate detection results in JSON format.
4. Send the generated JSON to FastAPI.
5. Receive detection results through an API endpoint.
6. Validate the received JSON structure and values.
7. Store validated detection events in PostgreSQL.
8. Store the complete original JSON response in a JSONB column.
9. Provide APIs to retrieve stored detection events.
10. Keep the backend architecture reusable for NVIDIA DeepStream integration.

---

## 3. Current Technology Stack

### AI / Computer Vision

- YOLO11n
- Ultralytics
- OpenCV
- PyTorch
- NumPy

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- Requests

### Database

- PostgreSQL 18
- psycopg2-binary
- JSONB

### Development Tools

- Visual Studio Code
- pgAdmin
- Git
- GitHub

### Future AI Pipeline

- NVIDIA DeepStream
- NVIDIA GPU
- TensorRT

---

## 4. Project Architecture

Current architecture:

5 Videos
    ↓
YOLO Object Detection
    ↓
Detection JSON
    ↓
send_json.py
    ↓
FastAPI
    ↓
Pydantic Validation
    ↓
PostgreSQL
    ↓
detection_events
    ↓
json_data JSONB


Future architecture:

5 Videos
    ↓
NVIDIA DeepStream
    ↓
AI Object Detection
    ↓
Detection JSON
    ↓
FastAPI
    ↓
Pydantic Validation
    ↓
PostgreSQL
    ↓
detection_events
    ↓
json_data JSONB

The FastAPI and PostgreSQL components are designed to remain unchanged when the inference layer is migrated from YOLO to DeepStream.

---

## 5. Project Structure

DeepStream_FastAPI_Project/

    database/
        schema.sql

    fastapi_app/
        database.py
        main.py
        models.py
        requirements.txt
        schemas.py
        send_json.py

    json_output/

    videos/

    yolo/
        detector.py
        requirements.txt

    .gitignore
    README.md

Additional YOLO-generated output may be created during execution:

    runs/

The runs directory contains generated YOLO detection output and is excluded from Git using .gitignore.

---

## 6. Folder Description

### videos/

Contains the five input videos used for testing.

Example:

    sample-5s.mp4
    sample-10s.mp4
    sample-15s.mp4
    sample-20s.mp4
    sample-30s.mp4

---

### yolo/

Contains the YOLO-based video detection implementation.

Main file:

    detector.py

This module:

- Lists available videos.
- Allows video selection.
- Displays the original video.
- Runs YOLO object detection.
- Displays the detection output.
- Generates detection JSON.
- Displays JSON detection summaries.

---

### json_output/

Contains JSON files generated from the AI detection process.

Example:

    sample-5s.json
    sample-10s.json
    sample-15s.json
    sample-20s.json
    sample-30s.json

---

### fastapi_app/

Contains the backend API implementation.

Files:

    main.py
    schemas.py
    database.py
    models.py
    send_json.py
    requirements.txt

---

### database/

Contains PostgreSQL database scripts.

Main file:

    schema.sql

This script creates the detection_events table.

---

## 7. Input Videos

The project uses five sample videos for testing.

Example video set:

    sample-5s.mp4
    sample-10s.mp4
    sample-15s.mp4
    sample-20s.mp4
    sample-30s.mp4

Each video can be processed independently.

---

## 8. YOLO Object Detection

YOLO11n is currently used as the object detection model.

The model detects objects in each video frame.

The detection result contains information such as:

- Frame number
- Timestamp
- Object class
- Class ID
- Confidence
- Bounding box coordinates

Example detected object information:

    class_id
    class_name
    confidence
    bounding_box

The bounding box contains:

    x1
    y1
    x2
    y2

---

## 9. Installing YOLO Dependencies

Open PowerShell in the project directory.

Run:

    pip install -r yolo\requirements.txt

The YOLO requirements include the packages required for:

- Ultralytics
- PyTorch
- OpenCV
- NumPy

Verify YOLO installation:

    python -c "from ultralytics import YOLO; print('YOLO installation successful')"

Expected output:

    YOLO installation successful

---

## 10. Running the YOLO Detector

Run:

    python yolo\detector.py

The program provides a menu:

    ============================================================
                 YOLO VIDEO DETECTION PIPELINE
    ============================================================

    1. Select Video
    2. View Original Video
    3. Run YOLO Detection
    4. View Detection Output
    5. Generate Detection JSON
    6. View Detection JSON
    7. Exit

---

## 11. Selecting a Video

Choose:

    1. Select Video

The application displays the available videos.

Select one of the five videos.

Example:

    sample-15s.mp4

The selected video becomes the input for the subsequent detection operations.

---

## 12. Viewing the Original Video

Choose:

    2. View Original Video

The application opens the selected video.

The video window is resizable and automatically fits within the available screen area.

Press:

    Q

to close the video window.

---

## 13. Running YOLO Detection

Choose:

    3. Run YOLO Detection

The YOLO model processes the selected video frame by frame.

YOLO identifies objects and generates an annotated detection video.

The output is stored inside the YOLO runs directory.

Example:

    runs\detect\predict

or:

    runs\detect\predict-2

The exact directory depends on the number of detection runs already performed.

---

## 14. Viewing Detection Output

Choose:

    4. View Detection Output

The application searches for the latest YOLO detection output and opens the processed video.

The detection output contains bounding boxes and class labels generated by YOLO.

Press:

    Q

to close the detection video.

---

## 15. Generating Detection JSON

Choose:

    5. Generate Detection JSON

The application processes the selected video and generates structured detection information.

The JSON file is saved inside:

    json_output/

Example:

    json_output/sample-10s.json

---

## 16. Detection JSON Structure

The generated JSON contains video-level information and frame-level detection information.

Example structure:

    {
        "video_name": "sample-10s.mp4",
        "video_path": "...",
        "total_frames": 303,
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "generated_at": "...",
        "detections": [
            {
                "frame_number": 1,
                "timestamp_seconds": 0.0,
                "objects": [
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.91,
                        "bounding_box": {
                            "x1": 100.0,
                            "y1": 120.0,
                            "x2": 300.0,
                            "y2": 350.0
                        }
                    }
                ]
            }
        ]
    }

The exact values depend on the video and the YOLO detections.

---

## 17. JSON Fields

### Video-Level Fields

    video_name

Name of the processed video.

    video_path

Path of the input video.

    total_frames

Total number of frames in the video.

    fps

Frames per second of the video.

    width

Video width in pixels.

    height

Video height in pixels.

    generated_at

Timestamp at which the JSON was generated.

    detections

List containing frame-level detection results.

---

## 18. Frame-Level Detection Fields

Each frame contains:

    frame_number

The frame number.

    timestamp_seconds

Timestamp of the frame in seconds.

    objects

List of objects detected in the frame.

---

## 19. Object Detection Fields

Each detected object contains:

    class_id

Numeric ID of the detected class.

    class_name

Name of the detected object.

    confidence

Model confidence score.

The value must be between:

    0.0 and 1.0

    bounding_box

Coordinates of the detected object's bounding box.

Bounding box fields:

    x1
    y1
    x2
    y2

---

## 20. PostgreSQL Database

The project uses a separate PostgreSQL database:

    deepstream_detection

This database is independent of the existing Order Management database.

The Order Management database is not modified by this project.

---

## 21. PostgreSQL Connection

The FastAPI application connects to PostgreSQL using:

    psycopg2-binary

The database connection configuration uses:

    DB_HOST
    DB_PORT
    DB_NAME
    DB_USER
    DB_PASSWORD

The application connects to the PostgreSQL database and performs database operations using psycopg2.

---

## 22. Testing PostgreSQL Connection

Run:

    python fastapi_app\database.py

A successful connection displays information similar to:

    ============================================================
         POSTGRESQL CONNECTION SUCCESSFUL
    ============================================================

    Host     : localhost
    Port     : 5432
    Database : deepstream_detection
    User     : postgres

    PostgreSQL Version:
    PostgreSQL 18.x

    ============================================================

---

## 23. Database Schema

The database contains a table named:

    detection_events

The table stores individual AI detection events.

The schema is located at:

    database/schema.sql

---

## 24. detection_events Table

The table contains the following columns:

    id
    video_name
    frame_number
    detected_class
    confidence
    timestamp_seconds
    x1
    y1
    x2
    y2
    json_data
    created_at

---

## 25. Column Description

### id

Primary key of the detection event.

### video_name

Name of the source video.

### frame_number

Frame in which the object was detected.

### detected_class

Detected object class.

Examples:

    person
    car
    bus
    truck

### confidence

Confidence score returned by the AI model.

### timestamp_seconds

Timestamp of the detection within the video.

### x1

Top-left or first horizontal bounding box coordinate.

### y1

Top-left or first vertical bounding box coordinate.

### x2

Bottom-right or second horizontal bounding box coordinate.

### y2

Bottom-right or second vertical bounding box coordinate.

### json_data

Complete AI detection JSON stored using PostgreSQL JSONB.

### created_at

Database timestamp indicating when the event was stored.

---

## 26. Why JSONB Is Used

The complete AI response is stored in:

    json_data

The column uses PostgreSQL:

    JSONB

This allows the system to retain the original structured detection response instead of storing only individual extracted fields.

This provides two forms of storage:

1. Structured columns for easy querying.
2. Complete JSONB data for preserving the original AI response.

---

## 27. Creating the Database Table

Open pgAdmin.

Connect to PostgreSQL.

Select:

    deepstream_detection

Open Query Tool.

Run the SQL script:

    database/schema.sql

After successful execution, the table should appear as:

    Tables
        detection_events

---

## 28. FastAPI Dependencies

Install the backend dependencies using:

    pip install -r fastapi_app\requirements.txt

The backend requires packages including:

    fastapi
    uvicorn
    pydantic
    psycopg2-binary
    requests

---

## 29. FastAPI Application

The main FastAPI application is:

    fastapi_app/main.py

The application provides endpoints for:

    GET /
    GET /health
    POST /detection
    GET /detection-events
    GET /detection-events/{event_id}

---

## 30. Starting FastAPI

From the project root directory, run:

    uvicorn fastapi_app.main:app --reload

The server starts at:

    http://127.0.0.1:8000

---

## 31. Swagger Documentation

Open:

    http://127.0.0.1:8000/docs

Swagger provides an interactive interface for testing the API.

---

## 32. ReDoc Documentation

Open:

    http://127.0.0.1:8000/redoc

ReDoc provides an alternative API documentation interface.

---

## 33. Root Endpoint

Endpoint:

    GET /

Purpose:

Checks whether the FastAPI application is running.

Example response:

    {
        "message": "DeepStream FastAPI Detection API is running"
    }

---

## 34. Health Endpoint

Endpoint:

    GET /health

Purpose:

Checks both the API and PostgreSQL connectivity.

This endpoint can be used to verify that the backend and database are available.

---

## 35. Detection Endpoint

Endpoint:

    POST /detection

Purpose:

Receives the complete AI detection JSON.

The endpoint performs the following operations:

    Receive JSON
        ↓
    Pydantic Validation
        ↓
    Extract Detection Events
        ↓
    Insert Events into PostgreSQL
        ↓
    Store Complete JSON
        ↓
    Return Success Response

---

## 36. JSON Validation

Pydantic schemas are defined in:

    fastapi_app/schemas.py

The schema validates:

- Required fields
- Data types
- Minimum values
- Maximum confidence values
- Bounding box structure
- Frame structure
- Object structure

For example, confidence must satisfy:

    0.0 <= confidence <= 1.0

Frame numbers must be valid positive values.

Required string fields cannot be empty.

---

## 37. Validation Flow

The incoming request is validated before database insertion.

The flow is:

    Incoming JSON
        ↓
    DetectionJSON
        ↓
    Pydantic Validation
        ↓
    Valid Data
        ↓
    PostgreSQL

If validation fails, FastAPI rejects the request instead of storing invalid detection data.

---

## 38. Database Storage Flow

For every detected object, the API extracts:

    video_name
    frame_number
    class_name
    confidence
    timestamp_seconds
    x1
    y1
    x2
    y2

These values are stored in individual database columns.

At the same time, the complete detection JSON is stored in:

    json_data

---

## 39. JSON Sending Client

The file:

    fastapi_app/send_json.py

is responsible for sending generated detection JSON to FastAPI.

Run:

    python fastapi_app\send_json.py

The application scans:

    json_output/

and displays available JSON files.

Select a JSON file to send it to:

    POST /detection

---

## 40. JSON Sending Process

The complete flow is:

    JSON file
        ↓
    send_json.py
        ↓
    HTTP POST
        ↓
    FastAPI /detection
        ↓
    Pydantic validation
        ↓
    PostgreSQL
        ↓
    Success response

---

## 41. End-to-End Test

The complete pipeline was successfully tested using:

    sample-10s.mp4

The generated JSON contained:

    Total frames: 303
    Total detected objects: 266

The JSON was sent using:

    python fastapi_app\send_json.py

FastAPI returned HTTP status:

    200

The response confirmed:

    Detection JSON received, validated and stored successfully

The response also reported:

    video_name: sample-10s.mp4
    total_frames: 303
    total_objects_received: 266
    events_inserted: 266
    validated: true
    stored: true
    database: deepstream_detection
    table: detection_events

This confirms the complete pipeline is functioning.

---

## 42. Event Storage

If the AI detects multiple objects across multiple frames, each detected object is stored as a separate event.

For example:

    Frame 1 → car
    Frame 1 → person
    Frame 2 → car
    Frame 3 → truck

These become separate rows in:

    detection_events

This makes individual detection events easy to query.

---

## 43. Retrieving Detection Events

All stored events can be retrieved using:

    GET /detection-events

A single event can be retrieved using:

    GET /detection-events/{event_id}

Example:

    GET /detection-events/1

This allows stored AI detection events to be accessed through the API.

---

## 44. Complete End-to-End Pipeline

The complete current implementation is:

    Video
       ↓
    YOLO11n
       ↓
    Object Detection
       ↓
    Detection JSON
       ↓
    send_json.py
       ↓
    HTTP POST
       ↓
    FastAPI
       ↓
    Pydantic Validation
       ↓
    Detection Event Extraction
       ↓
    PostgreSQL
       ↓
    detection_events
       ↓
    Individual Detection Rows
       +
    Complete JSONB Response

---

## 45. Example Data Flow

Suppose YOLO detects a car in frame 100.

The detection contains:

    class_name = car
    confidence = 0.91
    frame_number = 100
    timestamp_seconds = 3.33

The bounding box contains:

    x1
    y1
    x2
    y2

FastAPI validates this information.

Then PostgreSQL stores:

    video_name
    frame_number
    detected_class
    confidence
    timestamp_seconds
    x1
    y1
    x2
    y2

The complete JSON document is also stored in:

    json_data

---

## 46. Error Handling

The FastAPI application handles database errors using transaction management.

The database transaction follows:

    Begin
       ↓
    Insert events
       ↓
    Commit

If an error occurs:

    Rollback

This prevents incomplete database transactions from being committed.

---

## 47. Database Safety

The project uses a separate PostgreSQL database:

    deepstream_detection

The existing:

    order_management

database is not used by this project.

This separation prevents accidental modification of the Order Management project database.

---

## 48. Environment Variables

Database credentials should not be committed to GitHub.

Use environment variables for:

    DB_HOST
    DB_PORT
    DB_NAME
    DB_USER
    DB_PASSWORD

Example:

    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=deepstream_detection
    DB_USER=postgres
    DB_PASSWORD=YOUR_DATABASE_PASSWORD

The actual database password should never be placed directly in the public GitHub repository.

---

## 49. .gitignore

The project ignores files and directories that should not be committed.

Important ignored items include:

    __pycache__/
    *.pyc
    venv/
    .venv/
    runs/
    .vscode/
    .env
    .pytest_cache/

---

## 50. Running the Complete Project

### Step 1: Open the Project

Open the project in VS Code.

Project directory:

    DeepStream_FastAPI_Project

---

### Step 2: Install YOLO Dependencies

Run:

    pip install -r yolo\requirements.txt

---

### Step 3: Install FastAPI Dependencies

Run:

    pip install -r fastapi_app\requirements.txt

---

### Step 4: Configure PostgreSQL

Create the database:

    deepstream_detection

Make sure PostgreSQL is running.

---

### Step 5: Create the Detection Table

Open pgAdmin.

Select:

    deepstream_detection

Run:

    database/schema.sql

---

### Step 6: Test Database Connection

Run:

    python fastapi_app\database.py

Verify that the PostgreSQL connection is successful.

---

### Step 7: Start FastAPI

Run:

    uvicorn fastapi_app.main:app --reload

Keep this terminal running.

---

### Step 8: Open Swagger

Open:

    http://127.0.0.1:8000/docs

---

### Step 9: Generate YOLO Detection JSON

Open another terminal.

Run:

    python yolo\detector.py

Select:

    1. Select Video

Then:

    3. Run YOLO Detection

Then:

    5. Generate Detection JSON

---

### Step 10: Send JSON to FastAPI

Run:

    python fastapi_app\send_json.py

Select the generated JSON file.

---

### Step 11: Verify API Response

A successful request should return:

    HTTP Status: 200

and indicate:

    validated: true
    stored: true

---

### Step 12: Verify PostgreSQL

Open pgAdmin.

Navigate to:

    deepstream_detection
        ↓
    Schemas
        ↓
    public
        ↓
    Tables
        ↓
    detection_events

Use:

    SELECT * FROM detection_events;

to verify the stored detection events.

---

## 51. Testing Multiple Videos

The same process can be repeated for all five videos.

Example:

    sample-5s.mp4
    sample-10s.mp4
    sample-15s.mp4
    sample-20s.mp4
    sample-30s.mp4

Each video generates its own JSON file.

Example:

    json_output/
        sample-5s.json
        sample-10s.json
        sample-15s.json
        sample-20s.json
        sample-30s.json

Each JSON file can then be sent to FastAPI.

---

## 52. DeepStream Integration Plan

The current project uses YOLO because the development system does not have an NVIDIA GPU.

When an NVIDIA GPU is available, the inference layer will be migrated to NVIDIA DeepStream.

The future flow will be:

    Input Video
        ↓
    DeepStream Pipeline
        ↓
    NVIDIA GPU
        ↓
    AI Inference
        ↓
    Object Metadata
        ↓
    JSON Generation
        ↓
    FastAPI
        ↓
    Validation
        ↓
    PostgreSQL

The FastAPI API contract and PostgreSQL database design can remain reusable.

---

## 53. Why the Architecture Is Reusable

The project separates the AI inference layer from the backend layer.

Current inference:

    YOLO

Future inference:

    NVIDIA DeepStream

Both can produce the same logical detection information:

    video_name
    frame_number
    timestamp_seconds
    class_id
    class_name
    confidence
    bounding_box

Therefore, the backend does not need to depend on the specific AI inference engine.

---

## 54. Future DeepStream Components

When the NVIDIA GPU is available, the following technologies can be integrated:

- NVIDIA DeepStream
- TensorRT
- GStreamer
- NVIDIA GPU
- DeepStream inference plugins
- Metadata processing
- JSON output generation

The current FastAPI and PostgreSQL components can then receive the DeepStream-generated detection JSON.

---

## 55. Current Status

### Completed

- Five videos collected
- Project structure created
- YOLO installed
- YOLO11n tested
- Video detection implemented
- Detection output generated
- Detection JSON generated
- FastAPI application created
- Swagger documentation available
- Pydantic validation implemented
- PostgreSQL database created
- detection_events table created
- JSONB storage implemented
- JSON sending client implemented
- End-to-end API test completed
- Detection events successfully stored in PostgreSQL

### Next Phase

- NVIDIA GPU setup
- NVIDIA DeepStream installation
- DeepStream video pipeline
- DeepStream AI inference
- DeepStream detection metadata extraction
- DeepStream JSON generation
- Integration with the existing FastAPI backend
- End-to-end DeepStream → FastAPI → PostgreSQL testing

---

## 56. Important Project Principle

The project is intentionally divided into two major layers.

### AI Processing Layer

    YOLO / DeepStream
        ↓
    Detection JSON

### Backend Storage Layer

    Detection JSON
        ↓
    FastAPI
        ↓
    Validation
        ↓
    PostgreSQL

This makes it possible to change the AI inference technology without rebuilding the backend.

---

## 57. Troubleshooting

### FastAPI Module Import Error

If running the application from the project root, use:

    uvicorn fastapi_app.main:app --reload

Do not run the module as if all files were located directly in the root directory.

---

### PostgreSQL Connection Error

Check:

- PostgreSQL service is running.
- Database name is correct.
- Username is correct.
- Password is correct.
- Port is correct.
- Database is actually named deepstream_detection.

Test using:

    python fastapi_app\database.py

---

### YOLO Installation Error

Verify Python:

    python --version

Then install:

    pip install -r yolo\requirements.txt

Verify:

    python -c "from ultralytics import YOLO; print('YOLO installation successful')"

---

### FastAPI Not Opening

Check that Uvicorn is running:

    uvicorn fastapi_app.main:app --reload

Then open:

    http://127.0.0.1:8000/docs

---

### Invalid Detection JSON

If FastAPI rejects the JSON, verify that the generated JSON contains:

    video_name
    video_path
    total_frames
    fps
    width
    height
    generated_at
    detections

Each detection frame should contain:

    frame_number
    timestamp_seconds
    objects

Each object should contain:

    class_id
    class_name
    confidence
    bounding_box

---

## 58. GitHub

The project can be maintained using Git and GitHub.

Recommended repository name:

    DeepStream_FastAPI_Project

The repository should contain the source code, database schema, configuration templates, requirements, and documentation.

Generated files such as YOLO runs and Python cache files should remain excluded through .gitignore.

Large video files should be checked before pushing because GitHub has file-size limitations.

---

## 59. Git Commands

Initialize Git if required:

    git init

Check the repository:

    git status

Add project files:

    git add .

Create the first commit:

    git commit -m "Initial commit - DeepStream FastAPI detection pipeline"

Rename the branch:

    git branch -M main

Add the GitHub remote:

    git remote add origin YOUR_GITHUB_REPOSITORY_URL

Push the project:

    git push -u origin main

---

## 60. Security Before GitHub Push

Before pushing the project to GitHub, verify that sensitive information is not committed.

Do not commit:

    .env

Do not commit real:

    DB_PASSWORD

Do not expose database credentials in source code.

Use environment variables instead.

---

## 61. API Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | / | Check API status |
| GET | /health | Check API and database health |
| POST | /detection | Receive, validate and store detection JSON |
| GET | /detection-events | Retrieve detection events |
| GET | /detection-events/{event_id} | Retrieve a specific detection event |

---

## 62. Database Summary

Database:

    deepstream_detection

Table:

    detection_events

Primary key:

    id

Complete AI response:

    json_data JSONB

The table stores both structured detection information and the complete original JSON.

---

## 63. End-to-End Example

The complete execution can be summarized as:

    sample-10s.mp4
            ↓
        YOLO11n
            ↓
    303 video frames
            ↓
    Object detections
            ↓
    sample-10s.json
            ↓
       send_json.py
            ↓
    POST /detection
            ↓
    Pydantic Validation
            ↓
    266 detected objects
            ↓
    PostgreSQL
            ↓
    detection_events
            ↓
    266 stored events
            +
    Complete JSONB response

---

## 64. Future Production Architecture

The final intended architecture is:

    ┌───────────────────────┐
    │      5 Videos         │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │   NVIDIA DeepStream   │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │   AI Object Detection │
    │    NVIDIA GPU / TRT   │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │    Detection JSON     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │       FastAPI         │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │   Pydantic Validation  │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │      PostgreSQL       │
    │   detection_events    │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │      JSONB Data       │
    │     json_data         │
    └───────────────────────┘

---

## 65. Conclusion

DeepStream_FastAPI_Project provides an end-to-end foundation for AI video detection and event storage.

The current implementation uses YOLO for AI inference and successfully connects:

    Video
        ↓
    YOLO
        ↓
    JSON
        ↓
    FastAPI
        ↓
    Validation
        ↓
    PostgreSQL

The architecture is prepared for the next stage, where YOLO inference can be replaced with NVIDIA DeepStream and GPU-accelerated AI processing.

The backend API, validation layer, and PostgreSQL event-storage design are reusable for the future DeepStream implementation.

---

## 66. Author

Developed as a practical AI video analytics and backend integration project.

Technologies:

    Python
    YOLO
    FastAPI
    PostgreSQL
    NVIDIA DeepStream
    TensorRT
    OpenCV
    Git
    GitHub

---