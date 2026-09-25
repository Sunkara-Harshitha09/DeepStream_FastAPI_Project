from fastapi_app.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT COUNT(id)
    FROM detection_events
    WHERE video_name = %s
""", ("test_video - Trim 24s.mp4",))

count = cursor.fetchone()[0]

print("==============================================")
print(" PostgreSQL Detection Verification")
print("==============================================")
print("Video       :", "test_video - Trim 24s.mp4")
print("DB Records  :", count)
print("Expected    :", 4379)
print("==============================================")

cursor.close()
conn.close()