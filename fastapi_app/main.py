from fastapi import FastAPI, HTTPException
from psycopg2.extras import Json

from fastapi_app.schemas import DetectionJSON
from fastapi_app.database import get_connection


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Detection FastAPI",
    description=(
        "API for receiving, validating and storing "
        "YOLO / NVIDIA DeepStream detection results."
    ),
    version="1.0.0"
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Detection FastAPI is running",
        "status": "success"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            "SELECT 1;"
        )

        cursor.fetchone()

        cursor.close()

        return {
            "status": "healthy",
            "api": "running",
            "database": "connected",
            "database_name": "deepstream_detection"
        }

    except Exception as error:

        return {
            "status": "unhealthy",
            "api": "running",
            "database": "disconnected",
            "error": str(error)
        }

    finally:

        if connection is not None:
            connection.close()


# ============================================================
# RECEIVE, VALIDATE AND STORE DETECTION JSON
# ============================================================

@app.post("/detection")
def receive_detection(
    detection: DetectionJSON
):

    connection = None

    try:

        # ----------------------------------------------------
        # CONNECT TO POSTGRESQL
        # ----------------------------------------------------

        connection = get_connection()

        cursor = connection.cursor()

        # ----------------------------------------------------
        # COUNT DETECTIONS
        # ----------------------------------------------------

        total_objects = 0

        inserted_events = 0

        # ----------------------------------------------------
        # PROCESS EVERY FRAME
        # ----------------------------------------------------

        for frame in detection.detections:

            frame_number = frame.frame_number

            timestamp_seconds = (
                frame.timestamp_seconds
            )

            # ------------------------------------------------
            # PROCESS EVERY OBJECT IN FRAME
            # ------------------------------------------------

            for obj in frame.objects:

                total_objects += 1

                class_name = obj.class_name

                confidence = obj.confidence

                bounding_box = obj.bounding_box

                # --------------------------------------------
                # INSERT DETECTION INTO DATABASE
                # --------------------------------------------

                insert_query = """
                    INSERT INTO detection_events (
                        video_name,
                        frame_number,
                        detected_class,
                        confidence,
                        timestamp_seconds,
                        x1,
                        y1,
                        x2,
                        y2,
                        json_data
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    );
                """

                cursor.execute(
                    insert_query,
                    (
                        detection.video_name,

                        frame_number,

                        class_name,

                        confidence,

                        timestamp_seconds,

                        bounding_box.x1,

                        bounding_box.y1,

                        bounding_box.x2,

                        bounding_box.y2,

                        # Store the COMPLETE JSON response
                        Json(
                            detection.model_dump()
                        )
                    )
                )

                inserted_events += 1

        # ----------------------------------------------------
        # COMMIT TRANSACTION
        # ----------------------------------------------------

        connection.commit()

        cursor.close()

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "message": (
                "Detection JSON received, "
                "validated and stored successfully"
            ),

            "video_name":
                detection.video_name,

            "total_frames":
                detection.total_frames,

            "total_objects_received":
                total_objects,

            "events_inserted":
                inserted_events,

            "validated":
                True,

            "stored":
                True,

            "database":
                "deepstream_detection",

            "table":
                "detection_events"
        }

    except Exception as error:

        # ----------------------------------------------------
        # ROLLBACK IF SOMETHING FAILS
        # ----------------------------------------------------

        if connection is not None:

            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Detection data could not be stored: "
                f"{error}"
            )
        )

    finally:

        if connection is not None:

            connection.close()


# ============================================================
# GET ALL DETECTION EVENTS
# ============================================================

@app.get("/detection-events")
def get_detection_events():

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                video_name,
                frame_number,
                detected_class,
                confidence,
                timestamp_seconds,
                x1,
                y1,
                x2,
                y2,
                json_data,
                created_at
            FROM detection_events
            ORDER BY id;
            """
        )

        rows = cursor.fetchall()

        cursor.close()

        events = []

        for row in rows:

            events.append({

                "id":
                    row[0],

                "video_name":
                    row[1],

                "frame_number":
                    row[2],

                "detected_class":
                    row[3],

                "confidence":
                    float(row[4]),

                "timestamp_seconds":
                    float(row[5]),

                "x1":
                    float(row[6])
                    if row[6] is not None
                    else None,

                "y1":
                    float(row[7])
                    if row[7] is not None
                    else None,

                "x2":
                    float(row[8])
                    if row[8] is not None
                    else None,

                "y2":
                    float(row[9])
                    if row[9] is not None
                    else None,

                "json_data":
                    row[10],

                "created_at":
                    row[11]
            })

        return {
            "count": len(events),
            "events": events
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve detection events: "
                f"{error}"
            )
        )

    finally:

        if connection is not None:

            connection.close()


# ============================================================
# GET SINGLE DETECTION EVENT
# ============================================================

@app.get("/detection-events/{event_id}")
def get_detection_event(
    event_id: int
):

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                video_name,
                frame_number,
                detected_class,
                confidence,
                timestamp_seconds,
                x1,
                y1,
                x2,
                y2,
                json_data,
                created_at
            FROM detection_events
            WHERE id = %s;
            """,
            (event_id,)
        )

        row = cursor.fetchone()

        cursor.close()

        if row is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Detection event "
                    f"{event_id} not found"
                )
            )

        return {

            "id":
                row[0],

            "video_name":
                row[1],

            "frame_number":
                row[2],

            "detected_class":
                row[3],

            "confidence":
                float(row[4]),

            "timestamp_seconds":
                float(row[5]),

            "x1":
                float(row[6])
                if row[6] is not None
                else None,

            "y1":
                float(row[7])
                if row[7] is not None
                else None,

            "x2":
                float(row[8])
                if row[8] is not None
                else None,

            "y2":
                float(row[9])
                if row[9] is not None
                else None,

            "json_data":
                row[10],

            "created_at":
                row[11]
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve detection event: "
                f"{error}"
            )
        )

    finally:

        if connection is not None:

            connection.close()