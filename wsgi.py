from app import app  # Import the Flask instance from app.py

if __name__ == "__main__":
    # Ensure the app runs in production mode
    app.run(host="0.0.0.0", port=8000)  # Optional: Adjust the host and port if needed
