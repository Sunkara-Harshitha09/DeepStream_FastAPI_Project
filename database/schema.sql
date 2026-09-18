-- ============================================================
-- DEEPSTREAM DETECTION DATABASE
-- ============================================================

-- Table for storing AI detection events
-- Database: deepstream_detection


-- ============================================================
-- DETECTION EVENTS TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS detection_events (

    -- Unique database ID
    id SERIAL PRIMARY KEY,

    -- Name of the source video
    video_name VARCHAR(255) NOT NULL,

    -- Frame number where detection occurred
    frame_number INTEGER NOT NULL,

    -- Detected object/class name
    detected_class VARCHAR(100) NOT NULL,

    -- Detection confidence from YOLO/DeepStream
    confidence DECIMAL(6, 5) NOT NULL,

    -- Detection timestamp in seconds
    timestamp_seconds DECIMAL(12, 3) NOT NULL,

    -- Bounding box coordinates
    x1 DECIMAL(12, 2),

    y1 DECIMAL(12, 2),

    x2 DECIMAL(12, 2),

    y2 DECIMAL(12, 2),

    -- Complete original AI detection JSON
    json_data JSONB NOT NULL,

    -- Time when event was stored
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- VERIFY TABLE
-- ============================================================

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