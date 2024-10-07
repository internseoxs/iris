from flask import Flask, request, jsonify, render_template
import openai
import psycopg2
import json
import sqlparse
import re
from decimal import Decimal
from datetime import datetime, date
import logging
import os  # Import os to access environment variables

app = Flask(_name_)

# Initialize conversation history
conversation_history = []

# Configure logging to output to the terminal
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load configuration from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Check if any of the required environment variables are missing
if not openai.api_key:
    logging.error('OpenAI API key is missing from environment variables')
    raise ValueError('OpenAI API key is missing from environment variables')

if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
    logging.error('Database credentials are missing from environment variables')
    raise ValueError('Database credentials are missing from environment variables')

def get_database_schema():
    """Retrieve table and column names from the PostgreSQL database."""
    logging.debug("Attempting to retrieve database schema")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
        """)
        schema = cursor.fetchall()
        cursor.close()
        conn.close()

        schema_dict = {}
        for table, column in schema:
            if table not in schema_dict:
                schema_dict[table] = []
            schema_dict[table].append(column)

        logging.info("Successfully retrieved database schema")
        return schema_dict

    except Exception as e:
        logging.error("Error fetching database schema: %s", e)
        return {}


def format_schema_for_gpt(schema):
    """Format the schema in a way that can be included in the prompt to GPT."""
    logging.debug("Formatting database schema for GPT")
    formatted_schema = ""
    for table, columns in schema.items():
        formatted_schema += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
    return formatted_schema


@app.route("/", methods=["GET"])
def index():
    logging.info("Rendering index page")
    return render_template("index1.html")


@app.route("/submit_query", methods=["POST"])
def submit_query():
    prompt = request.form.get('prompt')
    if not prompt:
        logging.warning("No prompt provided in submit_query")
        return render_template("contact.html", query=None, results=None, error="No prompt provided.")
    logging.info("Received prompt in submit_query: %s", prompt)
    return render_template("about.html", prompt=prompt)


@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()
    if not data or 'prompt' not in data:
        logging.warning("No prompt provided in generate_response")
        return jsonify({'error': 'No prompt provided'}), 400

    prompt = data['prompt']
    logging.info("Received prompt in generate_response: %s", prompt)

    schema = get_database_schema()
    if not schema:
        logging.error("Failed to retrieve database schema")
        return jsonify({'error': 'Failed to retrieve database schema'}), 500

    formatted_schema = format_schema_for_gpt(schema)

    # Append the user's prompt to conversation history
    conversation_history.append({"role": "user", "content": prompt})
    logging.debug("Updated conversation history: %s", conversation_history)

    # Determine if new data is needed based on the current prompt
    if requires_more_data(prompt):
        logging.info("Determined that new data is required for the prompt")
        sql_query = generate_sql_query(prompt, formatted_schema, use_previous_response=True)
        if not sql_query:
            logging.error("Failed to generate SQL query")
            return jsonify({'error': 'Failed to generate SQL query'}), 500

        # Log the generated SQL query
        logging.info("Generated SQL query: %s", sql_query)

        new_data = execute_sql_query(sql_query)
        if new_data is None:
            logging.warning("SQL execution failed, proceeding with empty data")
            new_data = []  # Proceed with empty data if SQL execution failed

        # Store the new data in conversation history
        conversation_history.append({
            "role": "assistant",
            "content": f"Data Retrieved: {json.dumps(new_data, default=convert_to_serializable)}"
        })
        logging.debug("Data retrieved and added to conversation history")
    else:
        logging.info("No new data required for the prompt")
        new_data = None

    # Generate the final response
    final_response = generate_final_response(prompt, new_data, use_previous_response=True)
    if not final_response:
        logging.error("Failed to generate final response")
        return jsonify({'error': 'Failed to generate final response'}), 500

    # Append the assistant's response to conversation history
    conversation_history.append({"role": "assistant", "content": final_response})
    logging.debug("Assistant's response added to conversation history")

    logging.info("Returning final response to the user")
    return jsonify({'response': final_response})


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the conversation history and redirect to the home page."""
    global conversation_history
    conversation_history.clear()  # Clear the conversation history
    logging.info("Conversation history cleared via reset endpoint")
    return jsonify({"message": "Conversation history cleared."}), 200


def requires_more_data(prompt):
    """Determine if the prompt requires more data from the database."""
    logging.debug("Checking if more data is required for the prompt")
    # If no data has been retrieved yet, we need to fetch data
    data_retrieved = any(message for message in conversation_history if 'Data Retrieved' in message.get('content', ''))
    if not data_retrieved:
        logging.info("No data retrieved yet; need to fetch new data")
        return True  # Need to fetch new data
    print('==================================',conversation_history)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that determines if the user's query requires new data from the database. "
                "Respond 'yes' if new data is needed, otherwise respond 'no'."
            )
        }
    ] + conversation_history[-4:]  # Include the last few messages to provide context

    messages.append({
        "role": "user",
        "content": f"Does the following prompt require new data? {prompt}"
    })

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            n=1,
        )
        answer = response.choices[0].message['content'].strip().lower()
        logging.info("GPT response for requires_more_data: %s", answer)
        return answer == 'yes'
    except Exception as e:
        logging.error("Error determining if more data is needed: %s", e)
        return True  # Default to needing more data if GPT fails


def ensure_semicolon(sql_query):
    """Ensure the SQL query starts with SELECT and ends with a semicolon."""
    logging.debug("Ensuring SQL query is properly formatted")
    pattern = r"\bSELECT\b.*?(?=;|$)"
    match = re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL)
    if match:
        sql_query = match.group(0).strip()
        return sql_query + ';' if not sql_query.endswith(';') else sql_query
    return sql_query


def generate_sql_query(prompt, schema, use_previous_response=True):
    """Generate SQL query based on user prompt, schema, and previous responses."""
    logging.debug("Generating SQL query based on the prompt")
    messages = [
        {
            "role": "system",
            "content": (
                "Note that the current year is 2024. Generate the query with ILIKE keyword and use '%' sign before and after "
                "and do not use '=' whenever we use WHERE clause for datatype CHAR, VARCHAR, and TEXT. Use '=' for the rest of the cases. "
                "You are an assistant that converts user prompts into safe, read-only SQL queries. "
                "Here is the database schema:\n\n" + schema
            )
        }
    ]

    if use_previous_response:
        # Include the conversation history
        messages += conversation_history[-4:]  # Include the last few messages for context

    messages.append({
        "role": "user",
        "content": f"Generate an SQL query for the following prompt: {prompt}"
    })

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0,
            n=1,
        )
        sql_query = response.choices[0].message['content'].strip()
        logging.info("SQL query generated by GPT: %s", sql_query)
        return ensure_semicolon(sql_query)

    except Exception as e:
        logging.error("Error generating SQL query: %s", e)
        return None


def is_safe_sql(sql_query):
    """Check if the SQL query is safe and read-only."""
    logging.debug("Checking if the SQL query is safe")
    try:
        parsed = sqlparse.parse(sql_query)
        safe = all(statement.get_type() == 'SELECT' for statement in parsed)
        if not safe:
            logging.warning("SQL query is not safe: %s", sql_query)
        return safe
    except Exception as e:
        logging.error("Error parsing SQL query: %s", e)
        return False


def execute_sql_query(sql_query):
    """Execute the SQL query and return the result."""
    logging.debug("Executing SQL query")
    if not is_safe_sql(sql_query):
        logging.warning("Unsafe SQL query detected, not executing: %s", sql_query)
        return []  # Return an empty list to proceed with GPT response

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        logging.info("SQL query executed successfully")

        # Fetch limited results to prevent sending too much data
        results = cursor.fetchmany(500)  # Fetch up to 500 rows
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        logging.debug("Number of rows fetched: %d", len(results))
        return [dict(zip(colnames, row)) for row in results] if results else []
    except Exception as e:
        logging.error("Error executing SQL query: %s", e)
        return []  # Return an empty list to allow GPT to handle the response


def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable ones."""
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)  # Convert Decimal to float
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()  # Convert datetime/date to ISO 8601 string
    else:
        return obj


def chunk_data(data, chunk_size=100):
    """Split data into chunks of specified size."""
    logging.debug("Chunking data into sizes of %d", chunk_size)
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def generate_final_response(prompt, db_data, use_previous_response=True):
    """
    Generate the final response by passing the prompt and data to GPT.
    If no data is available from the database, GPT will answer directly.
    """
    logging.debug("Generating final response")
    messages = [
        {"role": "system", "content": "You are an assistant that helps users analyze data."}
    ]

    if use_previous_response:
        # Include conversation history for context
        messages += conversation_history[-6:]  # Include the last few messages
        logging.debug("Included conversation history in messages")

    if db_data:
        db_data_serializable = convert_to_serializable(db_data)
        data_chunks = list(chunk_data(db_data_serializable))
        logging.debug("Data has been serialized and chunked")

        # Due to token limits, we might need to process data in chunks
        final_response = []
        for idx, chunk in enumerate(data_chunks):
            logging.debug("Processing chunk %d/%d", idx + 1, len(data_chunks))
            temp_messages = messages.copy()
            temp_messages.append({
                "role": "user",
                "content": f"Prompt: {prompt}\nData: {json.dumps(chunk)}"
            })

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=temp_messages,
                    max_tokens=1500,
                    temperature=0.5,
                    n=1,
                )
                final_response.append(response.choices[0].message['content'].strip())
                logging.debug("Received response for chunk %d", idx + 1)

            except Exception as e:
                logging.error("Error generating GPT response: %s", e)
                return None

        logging.info("Successfully generated final response using data")
        return " ".join(final_response)

    else:
        # No new data; use previous conversation context
        messages.append({
            "role": "user",
            "content": f"Based on the previous data, {prompt}"
        })
        logging.debug("No new data available; generating response based on previous context")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1500,
                temperature=0.5,
                n=1,
            )
            logging.info("Successfully generated final response without new data")
            return response.choices[0].message['content'].strip()

        except Exception as e:
            logging.error("Error generating GPT response: %s", e)
            return None


if _name_ == "_main_":
    app.run(debug=True)
