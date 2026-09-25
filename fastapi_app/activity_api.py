import os
import json
import psycopg2

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List


app = FastAPI(
    title="People Activity API",
    version="1.0.0"
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv(
    "DB_NAME",
    "deepstream_detection"
)
DB_USER = os.getenv(
    "DB_USER",
    "postgres"
)
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Activity(BaseModel):

    activity: str

    start_time: float

    end_time: float

    duration_seconds: float


class Person(BaseModel):

    person_id: int

    activities: List[Activity]


class Summary(BaseModel):

    total_unique_persons: int

    walking: int

    standing: int

    sitting: int


class ActivityData(BaseModel):

    video_name: str

    video_duration_seconds: float

    summary: Summary

    persons: List[Person]


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    if not DB_PASSWORD:

        raise RuntimeError(
            "DB_PASSWORD environment variable is not set."
        )

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
            "People Activity API is running"
    }


# ============================================================
# ACTIVITY JSON API
# ============================================================

@app.post("/activity-json")
def receive_activity_data(
    data: ActivityData
):

    connection = None

    cursor = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Store complete JSON
        # ----------------------------------------------------

        activity_json = data.model_dump()

        # ----------------------------------------------------
        # Check whether this video already exists
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT id
            FROM person_activity_analysis
            WHERE video_name = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                data.video_name,
            )
        )

        existing = cursor.fetchone()

        if existing:

            # Update existing record
            cursor.execute(
                """
                UPDATE person_activity_analysis
                SET
                    video_duration_seconds = %s,
                    total_unique_persons = %s,
                    walking_count = %s,
                    standing_count = %s,
                    sitting_count = %s,
                    activity_json = %s::jsonb
                WHERE id = %s
                """,
                (
                    data.video_duration_seconds,
                    data.summary.total_unique_persons,
                    data.summary.walking,
                    data.summary.standing,
                    data.summary.sitting,
                    json.dumps(activity_json),
                    existing[0]
                )
            )

            record_id = existing[0]

            action = "updated"

        else:

            # Insert new record
            cursor.execute(
                """
                INSERT INTO person_activity_analysis
                (
                    video_name,
                    video_duration_seconds,
                    total_unique_persons,
                    walking_count,
                    standing_count,
                    sitting_count,
                    activity_json
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s::jsonb
                )
                RETURNING id
                """,
                (
                    data.video_name,
                    data.video_duration_seconds,
                    data.summary.total_unique_persons,
                    data.summary.walking,
                    data.summary.standing,
                    data.summary.sitting,
                    json.dumps(activity_json)
                )
            )

            record_id = cursor.fetchone()[0]

            action = "inserted"

        connection.commit()

        return {
            "message":
                "Activity data stored successfully",

            "action":
                action,

            "id":
                record_id,

            "video_name":
                data.video_name,

            "total_unique_persons":
                data.summary.total_unique_persons,

            "walking":
                data.summary.walking,

            "standing":
                data.summary.standing,

            "sitting":
                data.summary.sitting
        }

    except Exception as e:

        if connection:

            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=
                f"Activity data could not be stored: {str(e)}"
        )

    finally:

        if cursor:

            cursor.close()

        if connection:

            connection.close()