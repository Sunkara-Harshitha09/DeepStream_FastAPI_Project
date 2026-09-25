import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

# PostgreSQL configuration
# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "deepstream_detection")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_connection():
    """
    Create and return a PostgreSQL database connection.
    """

    if not DB_PASSWORD:
        raise ValueError(
            "DB_PASSWORD environment variable is not set."
        )

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    return connection


def test_database_connection():
    """
    Test the PostgreSQL database connection.
    """

    connection = None

    try:
        print("\nConnecting to PostgreSQL...")

        connection = get_connection()

        cursor = connection.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute("SELECT version();")

        result = cursor.fetchone()

        cursor.close()

        print("\n" + "=" * 60)
        print("     POSTGRESQL CONNECTION SUCCESSFUL")
        print("=" * 60)

        print(f"\nHost     : {DB_HOST}")
        print(f"Port     : {DB_PORT}")
        print(f"Database : {DB_NAME}")
        print(f"User     : {DB_USER}")

        print(
            f"\nPostgreSQL Version:\n{result['version']}"
        )

        print("\n" + "=" * 60)

    except psycopg2.Error as error:

        print("\n" + "=" * 60)
        print("     POSTGRESQL CONNECTION FAILED")
        print("=" * 60)

        print(f"\nError:\n{error}")

        print("\n" + "=" * 60)

    except ValueError as error:

        print("\n" + "=" * 60)
        print("     DATABASE CONFIGURATION ERROR")
        print("=" * 60)

        print(f"\nError:\n{error}")

        print("\n" + "=" * 60)

    finally:

        if connection is not None:
            connection.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    test_database_connection()