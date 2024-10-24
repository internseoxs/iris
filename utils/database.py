import psycopg2
import json
import logging
from psycopg2 import pool

# Load configuration from config.json file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Database connection parameters from config file
chat_history_db_config = config['database']['chat_history']

# Create connection pool for the chat history database
try:
    chat_history_pool = psycopg2.pool.SimpleConnectionPool(
        1,   # Minimum number of connections
        20,  # Maximum number of connections
        host=chat_history_db_config['host'],
        port=chat_history_db_config['port'],
        dbname=chat_history_db_config['dbname'],
        user=chat_history_db_config['user'],
        password=chat_history_db_config['password']
    )
    logging.info("Chat history connection pool created successfully")
except Exception as e:
    logging.error("Error creating database connection pool: %s", e)
    raise

# Function to get user details from the database
def get_user_by_username(username):
    try:
        conn = chat_history_pool.getconn()  # Use the chat_history connection pool
        cursor = conn.cursor()
        
        # Query to fetch user data based on the username
        cursor.execute("SELECT username, password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        cursor.close()
        chat_history_pool.putconn(conn)
        return user  # (username, password_hash)
    except Exception as e:
        logging.error("Error fetching user: %s", e)
        return None

# Function to add a new user to the database (run this from the backend)
def add_user(username, password_hash):
    try:
        conn = chat_history_pool.getconn()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (username, password_hash)
            VALUES (%s, %s)
        """, (username, password_hash))
        
        conn.commit()
        cursor.close()
        chat_history_pool.putconn(conn)
        print(f"User '{username}' added successfully.")
    except Exception as e:
        logging.error("Error adding user: %s", e)
