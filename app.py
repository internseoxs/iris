from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import openai
import psycopg2
import json
import sqlparse
import re
from decimal import Decimal
from datetime import datetime, date
import logging
import os
import uuid
from functools import wraps

app = Flask(__name__)

# Initialize conversation history
conversation_history = []

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from config.json file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Set OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logging.error('OpenAI API key is missing in the environment variables')
    raise ValueError('OpenAI API key is missing in environment variables')

# Secret key for Flask (used for sessions)
app.secret_key = config.get('secret_key')

# Database connection parameters from config file
sabre_db_config = config['database']['sabre_db1']
chat_history_db_config = config['database']['chat_history']


# Function to protect routes that require login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            logging.debug("User not logged in, redirecting to login page")
            return redirect(url_for('login'))
        logging.debug("User is logged in")
        return f(*args, **kwargs)

    return decorated_function


# Route to render the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Clear session every time login page is visited
    session.clear()

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = get_user_by_username(username)
        if user and check_password(password, user['password']):
            # Store user in session only after successful authentication
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template('login.html')


# Redirect any initial URL access to the login page
@app.route('/')
def home():
    # Always redirect to login page if user is not authenticated
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('index'))


# Route to logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


# Route to display the main chat interface, protected by login
@app.route('/index')
@login_required
def index():
    logging.info("Rendering index page")
    return render_template("index.html")


# Route to fetch chat history
@app.route("/chat-history", methods=["GET"])
@login_required
def chat_history_route():
    """Endpoint to fetch the chat history."""
    logging.info("Fetching chat history")
    return jsonify({'history': conversation_history})


# Route to submit a query
@app.route("/submit_query", methods=["POST"])
@login_required
def submit_query():
    prompt = request.form.get('prompt')
    if not prompt:
        logging.warning("No prompt provided in submit_query")
        return render_template("contact.html", query=None, results=None, error="No prompt provided.")
    logging.info("Received prompt in submit_query: %s", prompt)
    return render_template("about.html", prompt=prompt)


# Route to edit a prompt
@app.route('/edit-prompt', methods=['POST'])
@login_required
def edit_prompt():
    data = request.get_json()
    chat_index = data.get('chat_index')
    message_index = data.get('message_index')
    new_content = data.get('content')

    if chat_index is None or message_index is None or new_content is None:
        logging.warning("Incomplete data provided for editing prompt")
        return jsonify({'error': 'Incomplete data provided'}), 400

    try:
        if not (0 <= chat_index < len(conversation_history)):
            logging.error("Invalid chat index provided")
            return jsonify({'error': 'Invalid chat index'}), 400

        chat_session = conversation_history[chat_index]
        messages = chat_session['messages']

        if not (0 <= message_index < len(messages)):
            logging.error("Invalid message index provided")
            return jsonify({'error': 'Invalid message index'}), 400

        # Update the message
        messages[message_index]["content"] = new_content
        logging.info("Prompt updated successfully")

        # Remove assistant's response after the prompt
        if message_index + 1 < len(messages) and messages[message_index + 1]["role"] == "assistant":
            del messages[message_index + 1]

        return generate_response_for_edit(chat_index, message_index)
    except IndexError as e:
        logging.error("Error updating conversation history: %s", e)
        return jsonify({'error': 'An error occurred while editing the prompt'}), 500



# Route to generate a response
@app.route('/generate-response', methods=['POST'])
@login_required
def generate_response():
    data = request.get_json()
    if not data or 'prompt' not in data:
        logging.warning("No prompt provided")
        return jsonify({'error': 'No prompt provided'}), 400

    user_input = data['prompt']
    logging.info("Received prompt: %s", user_input)

    if is_incomplete_prompt(user_input):
        return jsonify({'error': 'Incomplete prompt provided'}), 400

    schema = get_database_schema()
    if not schema:
        return jsonify({'error': 'Failed to retrieve database schema'}), 500

    formatted_schema = format_schema_for_gpt(schema)

    if not conversation_history or 'reset' in data:
        conversation_history.clear()
        conversation_history.append({'messages': [], 'awaiting_clarification': False})

    chat_session = conversation_history[-1]
    messages = chat_session['messages']

    messages.append({"role": "user", "content": user_input})
    logging.debug("Updated conversation history: %s", messages)

    if chat_session.get('awaiting_clarification'):
        chat_session['awaiting_clarification'] = False
        logging.info("Processing clarification")
    else:
        clarification = clarify_prompt(user_input, messages)
        if clarification:
            messages.append({"role": "assistant", "content": clarification})
            chat_session['awaiting_clarification'] = True
            return jsonify({'response': clarification})

    previous_prompts = [msg['content'] for msg in messages if msg['role'] == 'user' and msg['content'] != user_input]
    is_related = is_related_to_previous_prompt(user_input, previous_prompts)

    requires_data = requires_more_data(user_input, messages)

    response = process_data_and_generate_response(chat_session, user_input, formatted_schema, messages, requires_data, is_related)
    return response

# Utility functions
def get_user_by_username(username):
    try:
        conn = psycopg2.connect(
            host=chat_history_db_config['host'],
            port=chat_history_db_config['port'],
            dbname=chat_history_db_config['dbname'],
            user=chat_history_db_config['user'],
            password=chat_history_db_config['password'],
            sslmode='require'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT username, password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            return {'username': user[0], 'password': user[1]}
        return None
    except Exception as e:
        logging.error("Error fetching user: %s", e)
        return None

def check_password(plain_password, stored_password):
    return plain_password == stored_password

# Additional utility functions such as get_database_schema, generate_sql_query, etc.
def get_database_schema():
    """Retrieve table and column names from the sabre_db1 PostgreSQL database."""
    logging.debug("Attempting to retrieve database schema from sabre_db1")
    try:
        conn = psycopg2.connect(
            host=sabre_db_config['host'],
            port=sabre_db_config['port'],
            dbname=sabre_db_config['dbname'],
            user=sabre_db_config['user'],
            password=sabre_db_config['password'],
            sslmode='require'
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

        logging.info("Successfully retrieved database schema from sabre_db1")
        return schema_dict

    except Exception as e:
        logging.error("Error fetching database schema from sabre_db1: %s", e)
        return {}

# Utility function examples
def is_incomplete_prompt(prompt):
    return len(prompt.strip()) == 0

def format_schema_for_gpt(schema):
    """Format the schema in a way that can be included in the prompt to GPT."""
    logging.debug("Formatting database schema for GPT")
    formatted_schema = ""
    for table, columns in schema.items():
        formatted_schema += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
    return formatted_schema

def get_serializable_messages(messages):
    """Extracts 'role' and 'content' from messages, ensures 'content' is serializable."""
    serializable_messages = []
    for msg in messages:
        serializable_msg = {
            "role": msg["role"],
            "content": msg["content"]
        }
        serializable_messages.append(serializable_msg)
    return serializable_messages

def clarify_prompt(prompt, messages):
    """
    Check if the prompt needs clarification and generate clarifying questions if needed.
    """
    logging.info("Checking if prompt needs clarification.")

    # System message to instruct GPT to ask clarifying questions if needed
    system_message = {
        "role": "system",
        "content": (
            "You are an assistant that checks if a user's prompt is clear enough to generate a SQL query. "
            "If the prompt is unclear, ask the user a clarifying question. "
            "If the prompt is clear, then provide the required answer."
        )
    }

    # Use the last few messages for context
    recent_messages = get_serializable_messages(messages[-4:]) if messages else []

    # Add the user's prompt to the conversation
    recent_messages.append({"role": "user", "content": f"Is this prompt clear: {prompt}"})

    try:
        # Call GPT to determine if clarification is needed
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[system_message] + recent_messages,
            max_tokens=50,
            temperature=0.0,
            n=1,
        )

        clarification_response = response.choices[0].message['content'].strip().lower()
        logging.info("GPT clarification check result: %s", clarification_response)

        # If GPT says "clear", return None (no clarification needed)
        if 'clear' in clarification_response:
            return None  # No clarification needed

        # Otherwise, return the clarification response as a question or guidance
        return clarification_response
    except Exception as e:
        logging.error("Error generating clarification question: %s", e)
        return None

def process_data_and_generate_response(chat_session, prompt, formatted_schema, messages, requires_data, is_related):
    """Process data and generate response based on the prompt, whether data is required, and if it's related to previous prompts."""
    max_attempts = 3  # Increased to allow more attempts for correct SQL generation
    attempt = 0
    sql_query = None
    new_data = None
    error_message = None

    if requires_data:
        previous_sql_query = None
        previous_data = None

        if is_related:
            # If the prompt is related to previous prompts, try to use previous SQL query and data
            for message in reversed(messages):
                if message['role'] == 'assistant' and 'sql_query' in message and 'data' in message:
                    previous_sql_query = message['sql_query']
                    previous_data = message['data']
                    break

        while attempt < max_attempts:
            attempt += 1
            if attempt == 1:
                # First attempt: Generate SQL query normally, using previous query/data if available
                sql_query = generate_sql_query(prompt, formatted_schema, messages, previous_sql_query, previous_data)
            else:
                # Subsequent attempts: Correct the SQL query based on the error
                if error_message:
                    sql_query = correct_sql_query(sql_query, error_message, formatted_schema, messages, previous_sql_query, previous_data)
                else:
                    break  # Cannot proceed without an error message

            if not sql_query:
                logging.error("Failed to generate SQL query on attempt %d", attempt)
                break

            # BEGIN: Validate the generated SQL query
            is_valid, validation_errors = validate_sql_query(sql_query, formatted_schema)
            if not is_valid:
                logging.error("SQL query validation failed on attempt %d: %s", attempt, validation_errors)
                error_message = f"SQL validation error: {validation_errors}"
                continue  # Attempt to regenerate or correct the SQL
            # END: Validate the generated SQL query

            logging.info("Attempting to execute SQL query: %s", sql_query)

            # Execute the SQL query
            new_data, error_message = execute_sql_query(sql_query)
            if new_data is not None:
                # Success
                # Do not include 'sql_query' and 'data' in messages passed to OpenAI API
                messages.append({
                    "role": "assistant",
                    "content": f"Data Retrieved."
                })
                # Store 'sql_query' and 'data' separately in the message, but these won't be sent to OpenAI API
                messages[-1]['sql_query'] = sql_query
                messages[-1]['data'] = new_data
                logging.debug("Data retrieved and added to conversation history")
                break
            else:
                logging.error("SQL query execution failed on attempt %d with error: %s", attempt, error_message)

        if new_data is None:
            # All attempts failed
            logging.error("All attempts to execute SQL query failed")
            # Let ChatGPT answer the prompt without data
            final_response = generate_final_response(prompt, None, messages)
            if not final_response:
                logging.error("Failed to generate final response")
                return jsonify({'error': 'Failed to generate final response'}), 500

            # Append the assistant's response to conversation history
            messages.append({"role": "assistant", "content": final_response})

            # Save to database
            save_interaction_to_db(prompt, final_response)
            # Note: Since no SQL query or data was used, we don't pass them here

            logging.info("Returning final response to the user")
            return jsonify({'response': final_response})
        else:
            # Generate the final response using the data retrieved
            final_response = generate_final_response(prompt, new_data, messages)
            if not final_response:
                logging.error("Failed to generate final response")
                return jsonify({'error': 'Failed to generate final response'}), 500

            # Append the assistant's response to conversation history
            messages.append({"role": "assistant", "content": final_response})

            # Save to database, including sql_query and data
            save_interaction_to_db(prompt, final_response, sql_query, new_data)

            logging.info("Returning final response to the user")
            return jsonify({'response': final_response})
    else:
        # If data is not required, generate final response without fetching data
        final_response = generate_final_response(prompt, None, messages)
        if not final_response:
            logging.error("Failed to generate final response")
            return jsonify({'error': 'Failed to generate final response'}), 500

        # Append the assistant's response to conversation history
        messages.append({"role": "assistant", "content": final_response})

        # Save to database
        save_interaction_to_db(prompt, final_response)
        # Note: Since no SQL query or data was used, we don't pass them here

        logging.info("Returning final response to the user")
        return jsonify({'response': final_response})

def validate_sql_query(sql_query, schema):
    """
    Validate the SQL query against the database schema.
    Returns a tuple (is_valid: bool, errors: str).
    """
    logging.debug("Validating SQL query against the database schema")

    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return False, "Failed to parse SQL query."

        statement = parsed[0]
        if statement.get_type() != 'SELECT':
            return False, "Only SELECT statements are allowed."

        # Extract tables and columns from the SQL query
        tables, columns = extract_tables_and_columns(sql_query)

        errors = []

        # Validate tables
        for table in tables:
            if table not in schema:
                errors.append(f"Table '{table}' does not exist in the database schema.")

        # Validate columns
        for table, cols in columns.items():
            if table not in schema:
                continue  # Table existence already checked
            for col in cols:
                if col not in schema[table]:
                    errors.append(f"Column '{col}' does not exist in table '{table}'.")

        if errors:
            return False, "; ".join(errors)

        return True, ""
    except Exception as e:
        logging.error("Error during SQL validation: %s", e)
        return False, str(e)

def extract_tables_and_columns(sql_query):
    """
    Extract table names and column names from the SQL query using sqlparse.
    Returns a tuple (tables: list, columns: dict).
    """
    logging.debug("Extracting tables and columns from SQL query")
    tables = []
    columns = {}

    try:
        parsed = sqlparse.parse(sql_query)
        stmt = parsed[0]

        select_seen = False

        for token in stmt.tokens:
            if token.ttype is sqlparse.tokens.DML and token.value.upper() == 'SELECT':
                select_seen = True
            if select_seen and isinstance(token, sqlparse.sql.IdentifierList):
                for identifier in token.get_identifiers():
                    if isinstance(identifier, sqlparse.sql.Identifier):
                        # Handle aliases
                        if identifier.get_real_name():
                            parent_name = identifier.get_parent_name()
                            if parent_name:
                                columns.setdefault(parent_name, []).append(identifier.get_real_name())
                            else:
                                # Handle cases without table aliases
                                columns.setdefault('unknown_table', []).append(identifier.get_real_name())
            if isinstance(token, sqlparse.sql.From):
                for identifier in token.get_sublists():
                    if isinstance(identifier, sqlparse.sql.IdentifierList):
                        for id in identifier.get_identifiers():
                            tables.append(id.get_real_name())
                    elif isinstance(identifier, sqlparse.sql.Identifier):
                        tables.append(identifier.get_real_name())
        return tables, columns
    except Exception as e:
        logging.error("Error extracting tables and columns: %s", e)
        return [], {}

def get_database_schema_dict():
    """Retrieve the database schema as a dictionary for validation."""
    return get_database_schema()

def format_schema_as_dict(schema):
    """Ensure the schema is in dictionary format."""
    return schema  # Assuming get_database_schema() already returns a dict

def generate_sql_query(prompt, schema, messages=None, previous_sql_query=None, previous_data=None):
    """Generate SQL query based on user prompt, schema, and conversation history."""
    logging.debug("Generating SQL query based on the prompt")

    # Create the base system message with instructions and schema
    system_prompt = (
        "As a SQL expert, generate a safe, read-only SQL query based on the user's prompt and the provided database schema. "
        "Generate correct syntax of query and only return SQL query in response. "
        "Determine what data is needed to answer the prompt. The query should fetch necessary data for analysis. "
        "Generate query in this way that if some names columns are present in that table then they must be extracted. "
        "If the effective date is greater than the ship date, then orders are late; if the effective date is less than or equal to the ship date, then it's on time. "
        "Ensure the query is syntactically correct and optimized. Do not ask for clarification.\n\n"
        "Database Schema:\n" + schema
    )

    if previous_sql_query and previous_data:
        # Include previous query and a sample of previous data
        sample_data = previous_data[:5] if len(previous_data) > 5 else previous_data
        system_prompt += (
            "\n\nPrevious SQL Query:\n" + previous_sql_query +
            "\n\nPrevious Data (first few rows):\n" + json.dumps(sample_data, default=convert_to_serializable)
        )

    messages_to_use = [{"role": "system", "content": system_prompt}]

    # Include the last few messages from the conversation history for context
    if messages:
        messages_to_use += get_serializable_messages(messages[-4:])

    # Append the user's prompt
    messages_to_use.append({
        "role": "user",
        "content": f"Generate an SQL query that retrieves the necessary data to answer the following analytical prompt. Do not include unnecessary data.\nPrompt: {prompt}"
    })

    try:
        # Call GPT to generate the SQL query
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages_to_use,
            max_tokens=500,
            temperature=0.0,
            n=1,
        )
        sql_query = response.choices[0].message['content'].strip()
        logging.info("SQL query generated by GPT: %s", sql_query)

        # Ensure the query is properly terminated with a semicolon
        sql_query = ensure_semicolon(sql_query)

        return sql_query

    except Exception as e:
        logging.error("Error generating SQL query: %s", e)
        return None

def correct_sql_query(original_query, error_message, schema, messages=None, previous_sql_query=None, previous_data=None):
    """Generate a corrected SQL query based on the error message."""
    logging.debug("Generating corrected SQL query based on error message")

    # Create the base system message with instructions and schema
    system_prompt = (
        "As a SQL expert, you have attempted to execute the following SQL query but encountered an error. "
        "Your task is to correct the SQL query based on the error message provided. Do not ask for clarification. "
        "Provide only the corrected SQL query.\n\n"
        "Original SQL Query:\n" + original_query +
        "\n\nError Message:\n" + error_message +
        "\n\nDatabase Schema:\n" + schema
    )

    if previous_sql_query and previous_data:
        # Include previous query and a sample of previous data
        sample_data = previous_data[:5] if len(previous_data) > 5 else previous_data
        system_prompt += (
            "\n\nPrevious SQL Query:\n" + previous_sql_query +
            "\n\nPrevious Data (first few rows):\n" + json.dumps(sample_data, default=convert_to_serializable)
        )

    messages_to_use = [{"role": "system", "content": system_prompt}]

    # Include the last few messages from the conversation history for context
    if messages:
        messages_to_use += get_serializable_messages(messages[-4:])

    try:
        # Call GPT to generate the corrected SQL query
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages_to_use,
            max_tokens=500,
            temperature=0.0,
            n=1,
        )
        corrected_sql_query = response.choices[0].message['content'].strip()
        logging.info("Corrected SQL query generated by GPT: %s", corrected_sql_query)

        # Ensure the query is properly terminated with a semicolon
        corrected_sql_query = ensure_semicolon(corrected_sql_query)

        return corrected_sql_query

    except Exception as e:
        logging.error("Error generating corrected SQL query: %s", e)
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
    """Execute the SQL query and return the result along with any error message."""
    logging.debug("Executing SQL query")
    if not is_safe_sql(sql_query):
        logging.warning("Unsafe SQL query detected, not executing: %s", sql_query)
        return None, "Unsafe SQL query detected."

    try:
        conn = psycopg2.connect(
            host=sabre_db_config['host'],
            port=sabre_db_config['port'],
            dbname=sabre_db_config['dbname'],
            user=sabre_db_config['user'],
            password=sabre_db_config['password'],
            sslmode='require'
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        logging.info("SQL query executed successfully")

        # Fetch limited results to prevent sending too much data
        results = cursor.fetchmany(1000)  # Fetch up to 1000 rows
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        logging.debug("Number of rows fetched: %d", len(results))
        data = [dict(zip(colnames, row)) for row in results] if results else []
        return data, None  # No error
    except Exception as e:
        logging.error("Error executing SQL query: %s", e)
        return None, str(e)  # Return error message

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

def generate_final_response(prompt, db_data, messages=None):
    """
    Generate the final response by passing the prompt and data to GPT.
    If no data is available from the database, GPT will answer directly.
    """
    logging.debug("Generating final response")
    messages_to_use = [
        {"role": "system",
         "content": "You are a data analyst assistant who provides detailed answers and insights based on the user's prompt and provided data. Use your analytical skills to interpret the data and provide actionable recommendations."}
    ]

    # Include the conversation history for context
    if messages:
        messages_to_use += get_serializable_messages(messages[-6:])  # Include the last few messages

    if db_data:
        # Process database data without chunking
        db_data_serializable = convert_to_serializable(db_data)
        logging.debug("Data has been serialized")

        # Send the entire data to GPT
        temp_messages = messages_to_use.copy()
        temp_messages.append({
            "role": "user",
            "content": (
                f"Based on the following data, {prompt}\n"
                f"Data: {json.dumps(db_data_serializable, default=convert_to_serializable)}"
            )
        })

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=temp_messages,
                max_tokens=8000,
                temperature=0.0,
                n=1,
            )
            final_response = response.choices[0].message['content'].strip()
            logging.info("Successfully generated final response using data")
            return final_response

        except Exception as e:
            logging.error("Error generating GPT response: %s", e)
            return None

    else:
        # No new data; generate response based on previous context
        temp_messages = messages_to_use.copy()
        temp_messages.append({
            "role": "user",
            "content": f"{prompt}"
        })
        logging.debug("No new data available; generating response based on the prompt")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=temp_messages,
                max_tokens=8000,
                temperature=0.0,
                n=1,
            )
            logging.info("Successfully generated final response without new data")
            final_response = response.choices[0].message['content'].strip()
            return final_response

        except Exception as e:
            logging.error("Error generating GPT response: %s", e)
            return None

def save_interaction_to_db(user_input, bot_response, sql_query=None, data=None):
    """Save the user input and bot response to the chat_history database."""
    try:
        conn = psycopg2.connect(
            host=chat_history_db_config['host'],
            port=chat_history_db_config['port'],
            dbname=chat_history_db_config['dbname'],
            user=chat_history_db_config['user'],
            password=chat_history_db_config['password'],
            sslmode='require'
        )
        cursor = conn.cursor()

        # Insert into chats table
        cursor.execute("""
            INSERT INTO chats (user_message, bot_response)
            VALUES (%s, %s)
            RETURNING id;
        """, (user_input, bot_response))
        chat_id = cursor.fetchone()[0]  # Get the ID of the inserted chat

        # Generate a UUID for the chat_id in response_history
        chat_uuid = str(uuid.uuid4())

        # Insert into response_history table, including sql_query and data if available
        cursor.execute("""
            INSERT INTO response_history (chat_id, user_input, bot_response, sql_query, data)
            VALUES (%s, %s, %s, %s, %s);
        """, (
            chat_uuid,
            user_input,
            bot_response,
            sql_query,
            json.dumps(data, default=convert_to_serializable) if data else None
        ))

        conn.commit()
        logging.info("Successfully stored chat with ID %s and response history with chat UUID %s", chat_id, chat_uuid)
    except Exception as e:
        logging.error("Error saving interaction to chat_history database: %s", e)
    finally:
        cursor.close()
        conn.close()

def requires_more_data(prompt, messages=None):
    """Determine if the prompt requires more data from the database."""
    logging.debug("Checking if more data is required for the prompt")

    if messages is None:
        # Fallback to global conversation history if no messages are passed
        messages = conversation_history[-1]['messages'] if conversation_history else []

    # Use GPT to determine if new data is needed
    messages_to_use = [
        {
            "role": "system",
            "content": (
                "As an AI assistant, determine if the user's prompt requires fresh data from the database to generate an accurate response. "
                "Respond with 'yes' if new data is needed or 'no' if it can be answered with previous data or without any data. "
                "Only respond with 'yes' or 'no'."
            )
        }
    ] + get_serializable_messages(messages[-4:])  # Include the last few messages for context

    messages_to_use.append({
        "role": "user",
        "content": f"Does the following prompt require new data? {prompt}"
    })

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages_to_use,
            max_tokens=5,  # Reduced to get concise 'yes' or 'no'
            temperature=0.0,
            n=1,
        )
        answer = response.choices[0].message['content'].strip().lower()
        logging.info("GPT response for requires_more_data: %s", answer)
        return answer == 'yes'
    except Exception as e:
        logging.error("Error determining if more data is needed: %s", e)
        return True  # Default to needing more data if GPT fails

def is_related_to_previous_prompt(new_prompt, previous_prompts):
    """Determine if the new prompt is related to the previous prompts."""
    logging.debug("Checking if the new prompt is related to previous prompts")

    if not previous_prompts:
        return False

    # Combine previous prompts into a single string
    previous_conversation = "\n".join(previous_prompts[-4:])  # Consider the last 4 prompts for context

    # Use GPT to analyze the relation between new_prompt and previous_prompts
    messages_to_use = [
        {
            "role": "system",
            "content": (
                "As an AI assistant, determine if the user's new prompt is related to the previous conversation. "
                "Respond with 'yes' if the new prompt is related or 'no' if it is not related. "
                "Only respond with 'yes' or 'no'."
            )
        },
        {
            "role": "user",
            "content": f"Previous conversation:\n{previous_conversation}\n\nNew prompt:\n{new_prompt}\n\nIs the new prompt related to the previous conversation?"
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages_to_use,
            max_tokens=5,  # Reduced to get concise 'yes' or 'no'
            temperature=0.0,
            n=1,
        )
        answer = response.choices[0].message['content'].strip().lower()
        logging.info("GPT response for is_related_to_previous_prompt: %s", answer)
        return answer == 'yes'
    except Exception as e:
        logging.error("Error determining if prompt is related to previous prompts: %s", e)
        return False  # Default to not related if GPT fails

def ensure_semicolon(sql_query):
    """Ensure the SQL query starts with SELECT and ends with a semicolon."""
    logging.debug("Ensuring SQL query is properly formatted")
    pattern = r"SELECT\b.*?(?=;|$)"
    match = re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL)
    if match:
        sql_query = match.group(0).strip()
        return sql_query + ';' if not sql_query.endswith(';') else sql_query
    return sql_query

# At the end of your code
if __name__ == "__main__":
    app.run(debug=True)
