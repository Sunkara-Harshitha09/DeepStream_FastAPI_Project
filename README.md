# DeepStream FastAPI Project

An end-to-end AI video analytics project combining YOLO-based computer vision, object detection, person detection and tracking, people activity analysis, FastAPI, PostgreSQL, Django, and MongoDB.

The project is designed so that the current YOLO inference layer can later be integrated with NVIDIA DeepStream when an NVIDIA GPU is available.

---

## 1. Project Overview

This project contains two main AI processing pipelines.

### Object Detection Pipeline

```text
Video
   ↓
YOLO11n
   ↓
Object Detection
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
```

### People Activity Pipeline

```text
People Activity Video
        ↓
YOLO Person Detection
        ↓
Person Tracking
        ↓
Activity Analysis
        ↓
People Activity JSON
        ↓
FastAPI
        ↓
PostgreSQL
        ↓
Django
        ↓
MongoDB
```

---

## 2. Project Objectives

The project implements the following:

- YOLO-based video object detection
- Person detection
- Person tracking
- People activity analysis
- Walking, standing, and sitting activity classification
- Structured JSON generation
- FastAPI APIs
- Pydantic JSON validation
- PostgreSQL storage
- PostgreSQL JSONB storage
- Django integration
- MongoDB storage
- PostgreSQL to MongoDB synchronization
- Scheduled synchronization
- Preparation for NVIDIA DeepStream integration

---

## 3. Technology Stack

### Computer Vision

- Python
- YOLO11n
- Ultralytics
- OpenCV
- PyTorch
- NumPy
- ByteTrack

### Backend

- FastAPI
- Uvicorn
- Pydantic
- Requests

### Databases

- PostgreSQL
- PostgreSQL JSONB
- MongoDB
- Django MongoDB Backend

### Development Tools

- Visual Studio Code
- PowerShell
- pgAdmin
- MongoDB Compass
- Docker
- Git
- GitHub

### Future GPU Pipeline

- NVIDIA DeepStream
- NVIDIA GPU
- TensorRT
- GStreamer

---

## 4. Project Structure

```text
DeepStream_FastAPI_Project/
│
├── database/
│   └── schema.sql
│
├── fastapi_app/
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   ├── send_json.py
│   ├── activity_api.py
│   ├── run_activity_api.py
│   └── requirements.txt
│
├── json_output/
│   └── detection JSON files
│
├── activity_output/
│   └── json/
│       └── people_activity_30s.json
│
├── yolo/
│   ├── detector.py
│   ├── activity_analyzer.py
│   ├── all_person_detector.py
│   ├── person_detector.py
│   ├── person_tracker.py
│   └── trackers/
│
├── videos/
│   └── input videos
│
├── check_detection_count.py
├── .gitignore
├── README.md
└── yolo11n.pt
```

Generated videos, detection output, tracking output, Python cache, environment files, and other large generated files are excluded through `.gitignore`.

---

## 5. Object Detection Pipeline

The object detection pipeline uses YOLO11n to process video frames.

For every detected object, the system can capture:

- Video name
- Frame number
- Timestamp
- Object class
- Confidence
- Bounding box

Bounding box coordinates:

```text
x1
y1
x2
y2
```

The detection results are converted into structured JSON.

---

## 6. Detection JSON

A detection JSON contains video information and frame-level detection information.

Example:

```json
{
  "video_name": "sample-10s.mp4",
  "total_frames": 303,
  "fps": 30.0,
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
```

---

## 7. PostgreSQL Detection Storage

The object detection pipeline stores individual detection events in PostgreSQL.

Database:

```text
deepstream_detection
```

Table:

```text
detection_events
```

The table contains:

```text
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
```

The complete original detection JSON is preserved in:

```text
json_data JSONB
```

This provides:

1. Individual structured fields for querying.
2. Complete original JSON for preserving the AI response.

---

## 8. FastAPI Detection API

The main FastAPI application is:

```text
fastapi_app/main.py
```

Main endpoints:

```text
GET  /
GET  /health
POST /detection
GET  /detection-events
GET  /detection-events/{event_id}
```

Start FastAPI:

```powershell
uvicorn fastapi_app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

## 9. Sending Detection JSON

The detection JSON sender is:

```text
fastapi_app/send_json.py
```

Run:

```powershell
python fastapi_app\send_json.py
```

The JSON is sent to:

```text
POST /detection
```

The processing flow is:

```text
Detection JSON
      ↓
FastAPI
      ↓
Pydantic Validation
      ↓
Detection Event Extraction
      ↓
PostgreSQL
      ↓
JSONB Storage
```

---

## 10. People Activity Analysis

A separate pipeline processes a people activity video.

Current video:

```text
people_activity_30s.mp4
```

The generated analysis uses a video duration of:

```text
23.79 seconds
```

The pipeline detects and tracks people and generates activity intervals.

Supported activities:

```text
walking
standing
sitting
```

---

## 11. People Activity JSON

The generated people activity JSON is:

```text
activity_output/json/people_activity_30s.json
```

Structure:

```json
{
  "video_name": "people_activity_30s.mp4",
  "video_duration_seconds": 23.79,
  "summary": {
    "total_unique_persons": 46,
    "walking": 6,
    "standing": 40,
    "sitting": 0
  },
  "persons": [
    {
      "person_id": 1,
      "activities": [
        {
          "activity": "standing",
          "start_time": 0.03,
          "end_time": 10.48,
          "duration_seconds": 10.44
        },
        {
          "activity": "walking",
          "start_time": 10.48,
          "end_time": 11.28,
          "duration_seconds": 0.8
        }
      ]
    }
  ]
}
```

The verified generated JSON contains:

```text
46 persons
281 activity records
```

---

## 12. People Activity PostgreSQL Storage

The people activity analysis is temporarily stored in PostgreSQL.

Table:

```text
person_activity_analysis
```

The table stores:

```text
id
video_name
video_duration_seconds
total_unique_persons
walking_count
standing_count
sitting_count
activity_json
created_at
```

The complete people activity JSON is stored in:

```text
activity_json JSONB
```

---

## 13. People Activity FastAPI API

The activity API is:

```text
fastapi_app/activity_api.py
```

Runner:

```text
fastapi_app/run_activity_api.py
```

Start the activity API:

```powershell
python fastapi_app\run_activity_api.py
```

The activity API runs on:

```text
http://127.0.0.1:8001
```

Main endpoint:

```text
POST /activity-json
```

---

## 14. Sending People Activity JSON

The complete people activity JSON can be sent to FastAPI using:

```powershell
Invoke-RestMethod `
    -Uri "http://127.0.0.1:8001/activity-json" `
    -Method Post `
    -ContentType "application/json" `
    -InFile ".\activity_output\json\people_activity_30s.json"
```

A successful response contains information such as:

```text
message              : Activity data stored successfully
action               : inserted
id                   : 1
video_name           : people_activity_30s.mp4
total_unique_persons : 46
walking              : 6
standing             : 40
sitting              : 0
```

---

## 15. Django + MongoDB Integration

The second project handles MongoDB synchronization.

Django project:

```text
Django_MongoDB
```

The Django application reads the people activity information from PostgreSQL and stores it in MongoDB.

MongoDB database:

```text
employee_management
```

MongoDB collection:

```text
employees_personactivityanalysis
```

---

## 16. MongoDB People Activity Structure

The complete people activity JSON is stored as one MongoDB document.

The document structure is:

```text
employees_personactivityanalysis
│
└── Document
    │
    ├── video_name
    ├── video_duration_seconds
    │
    ├── summary
    │   ├── total_unique_persons
    │   ├── walking
    │   ├── standing
    │   └── sitting
    │
    ├── persons
    │   ├── person 1
    │   │   └── activities
    │   ├── person 2
    │   │   └── activities
    │   ├── ...
    │   └── person 46
    │       └── activities
    │
    ├── postgres_created_at
    └── synced_at
```

The verified MongoDB document contains:

```text
46 persons
281 activities
```

---

## 17. Django Activity Model

The Django model represents the MongoDB activity document using:

```text
PersonActivityAnalysis
```

Main fields:

```text
video_name
video_duration_seconds
summary
persons
postgres_created_at
synced_at
```

The `summary` and `persons` fields preserve the nested JSON structure.

---

## 18. PostgreSQL to MongoDB Synchronization

The activity synchronization service is:

```text
employees/activity_sync_service.py
```

The synchronization flow is:

```text
PostgreSQL
     ↓
Read person_activity_analysis
     ↓
Read activity_json
     ↓
Extract summary
     ↓
Extract persons
     ↓
Create or update MongoDB document
```

The synchronization service uses batching.

Current batch size:

```text
100
```

---

## 19. Activity Scheduler

The activity scheduler is:

```text
run_activity_scheduler.py
```

The scheduler is configured for a two-minute synchronization interval.

Flow:

```text
Start Scheduler
      ↓
Sync PostgreSQL to MongoDB
      ↓
Display Result
      ↓
Wait 2 Minutes
      ↓
Sync Again
      ↓
Repeat
```

Start the scheduler:

```powershell
python run_activity_scheduler.py
```

Example scheduler output:

```text
PEOPLE ACTIVITY SYNC SCHEDULER

Source      : PostgreSQL
Destination : MongoDB
Interval    : 2 minutes
Status      : Running
```

Stop the scheduler:

```text
CTRL + C
```

---

## 20. Detection MongoDB Collection

Detection-event synchronization is separate from the people activity pipeline.

MongoDB collection:

```text
employees_detectionevent
```

The detection synchronization processes PostgreSQL records in batches.

Current batch size:

```text
500
```

This allows large detection datasets to be synchronized in manageable batches.

---

## 21. Detection Data Structure

Each detection document contains fields such as:

```text
postgres_id
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
postgres_created_at
synced_at
```

Multiple detections can exist for the same video frame.

Therefore:

```text
Number of MongoDB detection documents
does not equal
Number of video frames
```

A frame containing multiple detected objects can produce multiple detection records.

---

## 22. Database Separation

The project uses:

```text
deepstream_detection
```

for PostgreSQL video detection and activity processing.

This database is separate from the Order Management project database.

The Order Management database is not modified by this project.

---

## 23. Environment Variables

Database credentials must not be committed to GitHub.

Use environment variables such as:

```text
DB_HOST
DB_PORT
DB_NAME
DB_USER
DB_PASSWORD
```

Example:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=deepstream_detection
DB_USER=postgres
DB_PASSWORD=YOUR_DATABASE_PASSWORD
```

The real password must never be committed to the repository.

---

## 24. Git Ignore

Important ignored files and directories include:

```text
.env
__pycache__/
*.pyc
venv/
.venv/
runs/
.vscode/
.pytest_cache/
yolo11n-pose.pt
videos/*.mp4
person_detection_output/
person_tracking_output/
```

Large generated videos and processing outputs should remain outside GitHub.

---

## 25. Installation

From the project root:

```powershell
pip install -r yolo\requirements.txt
```

Install FastAPI dependencies:

```powershell
pip install -r fastapi_app\requirements.txt
```

Verify YOLO:

```powershell
python -c "from ultralytics import YOLO; print('YOLO installation successful')"
```

---

## 26. PostgreSQL Setup

Create the PostgreSQL database:

```text
deepstream_detection
```

Make sure PostgreSQL is running.

Create the detection table using:

```text
database/schema.sql
```

Test the PostgreSQL connection:

```powershell
python fastapi_app\database.py
```

---

## 27. Object Detection Workflow

```text
Input Video
     ↓
YOLO11n
     ↓
Object Detection
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
```

---

## 28. People Activity Workflow

```text
People Activity Video
        ↓
Person Detection
        ↓
Person Tracking
        ↓
Activity Analysis
        ↓
People Activity JSON
        ↓
FastAPI
        ↓
PostgreSQL
        ↓
Django Synchronization
        ↓
MongoDB
```

---

## 29. Current People Activity Result

The current verified people activity analysis contains:

```text
Video:
people_activity_30s.mp4

Duration:
23.79 seconds

Unique Persons:
46

Walking:
6

Standing:
40

Sitting:
0

Activity Records:
281
```

---

## 30. Current Detection Pipeline Result

The detection pipeline has been tested with multiple videos and stores individual detection events in PostgreSQL and MongoDB.

The detection MongoDB synchronization uses a batch size of 500 records.

This approach was used to process the large detection dataset without loading all records into memory at once.

---

## 31. Error Handling

The FastAPI services use database transactions.

General flow:

```text
Begin Transaction
       ↓
Validate Data
       ↓
Insert or Update
       ↓
Commit
```

If an error occurs:

```text
Rollback
```

This prevents incomplete database transactions from being committed.

---

## 32. Security

Never commit:

```text
.env
```

Never expose:

```text
DB_PASSWORD
SECRET_KEY
JWT tokens
MongoDB passwords
API keys
```

Use environment variables for sensitive configuration.

---

## 33. GitHub

Recommended repository name:

```text
DeepStream_FastAPI_Project
```

The GitHub repository should contain:

- Source code
- Database schema
- Requirements
- Documentation
- Configuration templates
- Appropriate JSON examples

The repository should not contain:

- Passwords
- `.env`
- Large videos
- Generated detection output
- Python cache
- Unnecessary large model files

---

## 34. Future NVIDIA DeepStream Integration

The current project uses YOLO because the development system does not have an NVIDIA GPU.

The planned future architecture is:

```text
Input Videos
     ↓
NVIDIA DeepStream
     ↓
NVIDIA GPU
     ↓
TensorRT
     ↓
AI Inference
     ↓
Detection Metadata
     ↓
JSON
     ↓
FastAPI
     ↓
PostgreSQL
```

Future technologies include:

- NVIDIA DeepStream
- TensorRT
- GStreamer
- NVIDIA GPU
- DeepStream inference plugins
- Metadata processing

---

## 35. Current Project Status

### Completed

- YOLO11n setup
- Video processing
- Object detection
- Detection JSON generation
- FastAPI detection API
- Pydantic validation
- PostgreSQL detection storage
- PostgreSQL JSONB storage
- Detection JSON sending client
- Person detection
- Person tracking
- People activity analysis
- Walking / standing / sitting analysis
- People activity JSON generation
- People activity PostgreSQL storage
- Django + MongoDB integration
- Complete people activity JSON stored in MongoDB
- MongoDB activity structure verified
- 46 people stored
- 281 activity records stored
- Two-minute scheduler implemented and tested

### Remaining Work

- Final verification of automatic two-minute activity synchronization
- Two-hour PostgreSQL cleanup
- MongoDB verification before PostgreSQL deletion
- End-to-end synchronization and cleanup testing
- NVIDIA GPU setup
- NVIDIA DeepStream integration
- TensorRT pipeline
- DeepStream to FastAPI integration
- DeepStream to PostgreSQL testing

---

## 36. Final Architecture

```text
                         VIDEO INPUT
                              │
                              ▼
                    ┌──────────────────┐
                    │     YOLO11n      │
                    └────────┬─────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
                   ▼                   ▼
            Object Detection     Person Detection
                   │                   │
                   ▼                   ▼
             Detection JSON      Person Tracking
                   │                   │
                   ▼                   ▼
                FastAPI          Activity Analysis
                   │                   │
                   ▼                   ▼
             PostgreSQL          Activity JSON
                   │                   │
                   │                   ▼
                   │              PostgreSQL
                   │                   │
                   │                   ▼
                   │                 Django
                   │                   │
                   │                   ▼
                   │                MongoDB
                   │
                   ▼
             detection_events
```

---

## 37. Project Principle

The project separates the AI processing layer from the backend and database layers.

### AI Processing Layer

```text
YOLO / DeepStream
        ↓
Detection / Activity Analysis
        ↓
JSON
```

### Backend Layer

```text
JSON
 ↓
FastAPI
 ↓
Validation
 ↓
PostgreSQL
```

### Synchronization Layer

```text
PostgreSQL
 ↓
Django
 ↓
MongoDB
```

This architecture allows the AI inference technology to evolve without rebuilding the complete backend architecture.

---

## 38. End-to-End Summary

### Object Detection

```text
Video
 ↓
YOLO11n
 ↓
Detection JSON
 ↓
FastAPI
 ↓
Validation
 ↓
PostgreSQL
 ↓
detection_events
 ↓
MongoDB detection collection
```

### People Activity

```text
People Activity Video
 ↓
Person Detection
 ↓
Person Tracking
 ↓
Activity Analysis
 ↓
people_activity_30s.json
 ↓
FastAPI
 ↓
PostgreSQL
 ↓
Django
 ↓
MongoDB
```

### Planned Cleanup

```text
PostgreSQL
 ↓
MongoDB Sync
 ↓
Verify MongoDB Copy
 ↓
After 2 Hours
 ↓
Delete PostgreSQL Record
 ↓
Keep MongoDB Data Permanently
```

---

## 39. Author

Developed as a practical AI video analytics, computer vision, backend integration, database, and synchronization project.

### Technologies

```text
Python
YOLO11n
Ultralytics
OpenCV
PyTorch
ByteTrack
FastAPI
PostgreSQL
JSONB
Django
MongoDB
Docker
NVIDIA DeepStream
TensorRT
Git
GitHub
```

---

## 40. Repository

Project:

```text
DeepStream_FastAPI_Project
```

Backend database:

```text
deepstream_detection
```

People activity MongoDB database:

```text
employee_management
```

People activity MongoDB collection:

```text
employees_personactivityanalysis
```

Detection MongoDB collection:

```text
employees_detectionevent
```