# ============================================================
# DATABASE MODEL INFORMATION
# ============================================================
#
# The actual PostgreSQL table is created using:
#
# database/schema.sql
#
# Table:
# detection_events
#
# This file keeps the database structure documented in the
# FastAPI application.
# ============================================================


DETECTION_EVENTS_TABLE = "detection_events"


DETECTION_EVENT_COLUMNS = [

    "id",

    "video_name",

    "frame_number",

    "detected_class",

    "confidence",

    "timestamp_seconds",

    "x1",

    "y1",

    "x2",

    "y2",

    "json_data",

    "created_at"
]