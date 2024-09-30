import psycopg2

def test_connection():
    try:
        connection = psycopg2.connect(
            host="localhost",  # or "127.0.0.1"
            database="sabre_db1",
            user="test",
            password="test"
        )
        print("Connection to PostgreSQL DB successful")
        connection.close()
    except Exception as e:
        print(f"The error '{e}' occurred")

test_connection()