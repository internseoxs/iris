from flask import Flask, request, jsonify, render_template
import openai
import psycopg2
import json
import sqlparse
import re
from decimal import Decimal
from datetime import datetime, date
import logging
import os  # To access environment variables

app = Flask(__name__)

# Initialize conversation history
conversation_history = []

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logging.error('OpenAI API key is missing in the environment variables')
    raise ValueError('OpenAI API key is missing in environment variables')

# Database connection parameters from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

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


@app.route('/edit-prompt', methods=['POST'])
def edit_prompt():
    data = request.get_json()
    chat_index = data.get('chat_index')  # Index of the chat session
    message_index = data.get('message_index')  # Index of the prompt to edit
    new_content = data.get('content')  # The updated prompt content

    if chat_index is None or message_index is None or new_content is None:
        logging.warning("Incomplete data provided for editing prompt")
        return jsonify({'error': 'Incomplete data provided'}), 400

    try:
        # Update the specific message's content in conversation_history
        chat_session = conversation_history[chat_index]
        messages = chat_session['messages']

        if 0 <= message_index < len(messages):
            # Replace the old prompt with the new content
            messages[message_index]["content"] = new_content
            logging.info("Successfully updated prompt in conversation history")

            # Remove the assistant's response that follows this message
            if message_index + 1 < len(messages) and messages[message_index + 1]["role"] == "assistant":
                del messages[message_index + 1]

            # Generate a new response
            return generate_response_for_edit(chat_index, message_index)
        else:
            logging.error("Invalid message index provided for editing prompt")
            return jsonify({'error': 'Invalid message index'}), 400

    except IndexError as e:
        logging.error("Error updating conversation history: %s", e)
        return jsonify({'error': 'An error occurred while editing the prompt'}), 500


def generate_response_for_edit(chat_index, message_index):
    """Generates a response for an edited prompt starting from generate_response workflow."""
    chat_session = conversation_history[chat_index]
    messages = chat_session['messages']

    # Extract the updated prompt
    prompt = messages[message_index]["content"]
    logging.info("Processing edited prompt in generate_response_for_edit: %s", prompt)

    # Remove any messages after the edited prompt (if any)
    del messages[message_index + 1:]

    schema = get_database_schema()
    if not schema:
        logging.error("Failed to retrieve database schema")
        return jsonify({'error': 'Failed to retrieve database schema'}), 500

    formatted_schema = format_schema_for_gpt(schema)

    # Check if new data is required based on the edited prompt
    if requires_more_data(prompt, messages):
        logging.info("New data required for edited prompt")
        sql_query = generate_sql_query(prompt, formatted_schema, messages)
        if not sql_query:
            logging.error("Failed to generate SQL query")
            return jsonify({'error': 'Failed to generate SQL query'}), 500

        # Execute the SQL query and retrieve data
        new_data = execute_sql_query(sql_query)
        if new_data is None:
            logging.warning("SQL execution failed, proceeding with empty data")
            new_data = []  # Proceed with empty data if SQL execution failed

        # Store the new data in conversation history as a new entry
        messages.append({
            "role": "assistant",
            "content": f"Data Retrieved: {json.dumps(new_data, default=convert_to_serializable)}"
        })
    else:
        logging.info("No new data required for edited prompt")
        new_data = None

    # Generate the final response based on the edited prompt
    final_response = generate_final_response(prompt, new_data, messages)
    if not final_response:
        logging.error("Failed to generate final response")
        return jsonify({'error': 'Failed to generate final response'}), 500

    # Insert the assistant's response directly after the edited prompt
    messages.insert(message_index + 1, {
        "role": "assistant",
        "content": final_response
    })
    logging.debug("Assistant's response to edited prompt added to conversation history")

    # Return the final response to the frontend
    return jsonify({'response': final_response})


def is_incomplete_prompt(prompt):
    # Example rule-based logic (you can customize this)
    if len(prompt.split()) < 5:  # If the prompt is less than 5 words
        return True
    if '?' not in prompt and len(prompt.split()) < 10:  # If the prompt has no questions and is too short
        return True
    # Add other checks as needed
    return False


@app.route("/", methods=["GET"])
def index():
    logging.info("Rendering index page")
    return render_template("final_index.html")


@app.route("/chat-history", methods=["GET"])
def chat_history_route():
    """Endpoint to fetch the chat history."""
    logging.info("Fetching chat history")
    return jsonify({'history': conversation_history})


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

    # Check if a reset is requested or no conversation history exists
    if not conversation_history or 'reset' in data:
        conversation_history.clear()  # Clear history if reset is requested
        conversation_history.append({'messages': []})

    chat_session = conversation_history[-1]  # Access the most recent chat session
    messages = chat_session['messages']

    # Append the user's prompt to conversation history
    messages.append({"role": "user", "content": prompt})
    logging.debug("Updated conversation history: %s", messages)

    # Determine if new data is needed based on the current prompt
    if requires_more_data(prompt, messages):
        logging.info("Determined that new data is required for the prompt")
        sql_query = generate_sql_query(prompt, formatted_schema, messages)
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
        messages.append({
            "role": "assistant",
            "content": f"Data Retrieved: {json.dumps(new_data, default=convert_to_serializable)}"
        })
        logging.debug("Data retrieved and added to conversation history")
    else:
        logging.info("No new data required for the prompt")
        new_data = None

    # Generate the final response
    final_response = generate_final_response(prompt, new_data, messages)
    if not final_response:
        logging.error("Failed to generate final response")
        return jsonify({'error': 'Failed to generate final response'}), 500

    # Append the assistant's response to conversation history
    messages.append({"role": "assistant", "content": final_response})
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


def requires_more_data(prompt, messages=None):
    """Determine if the prompt requires more data from the database."""
    logging.debug("Checking if more data is required for the prompt")

    # If no data has been retrieved yet, we need to fetch data
    if messages is None:
        # Fallback to global conversation history if no messages are passed
        messages = conversation_history[-1]['messages'] if conversation_history else []

    data_retrieved = any('Data Retrieved' in message.get('content', '') for message in messages)
    if not data_retrieved:
        logging.info("No data retrieved yet; need to fetch new data")
        return True  # Need to fetch new data

    # Use the last few messages to provide context
    messages_to_use = [
                          {
                              "role": "system",
                              "content": (
                                  "You are a conversational AI model that determines if the user's query requires new data from the database. "
                                  "Respond 'yes' if new data is needed, otherwise respond 'no'."
                              )
                          }
                      ] + messages[-4:]  # Include the last few messages to provide context

    messages_to_use.append({
        "role": "user",
        "content": f"Does the following prompt require new data? {prompt}"
    })

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages_to_use,
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


def generate_sql_query(prompt, schema, messages=None, use_previous_response=True):
    """Generate SQL query based on user prompt, schema, and conversation history."""
    logging.debug("Generating SQL query based on the prompt")

    # Create the base system message with instructions and schema
    messages_to_use = [
        {
            "role": "system",
            "content": (
                    "Note that the current year is 2024. Generate the query with ILIKE keyword and use '%' sign before and after "
                    "and do not use '=' whenever we use WHERE clause for datatype CHAR, VARCHAR, and TEXT. Use '=' for the rest of the cases. "
                    "You are an assistant that converts user prompts into safe, read-only SQL queries. "
                    " ship_date  is the date on which delievery was expected and  effective_date is the date on which the order is actuallu delievered."
                    "if effective date > ship order  which means delivery status is late otherwise on time"

                    "You also generate clarifying questions when necessary to understand the user's intent better."
                    " there  is column invoice_status in sale_order table  where three unique values are present.no invoice means that its state is cancelled ,invoiced means that state=done and to invoice means that state=draft."
                    "if require, ask some clarifying questions to generate a better response "
                    "Here is the database schema:\n\n" + schema
            )
        }
    ]

    # If no messages are passed, fall back to conversation history
    if messages is None and use_previous_response and conversation_history:
        messages = conversation_history[-1]['messages'] if conversation_history else []

    # Include the last few messages from the conversation history for context
    if messages:
        messages_to_use += messages[-4:]

    # Append the user's prompt
    messages_to_use.append({
        "role": "user",
        "content": f"Generate an SQL query for the following prompt: {prompt}"
    })

    try:
        # Call GPT to generate the SQL query
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages_to_use,
            max_tokens=1000,
            temperature=0,
            n=1,
        )
        sql_query = response.choices[0].message['content'].strip()
        logging.info("SQL query generated by GPT: %s", sql_query)

        # Ensure the query is properly terminated with a semicolon
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


def chunk_data(data, chunk_size=200):
    """Split data into chunks of specified size."""
    logging.debug("Chunking data into sizes of %d", chunk_size)
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


response_history = {'previous_response': []}


def generate_final_response(prompt, db_data, column_explanations=None, is_follow_up=False, use_previous_response=True):
    """
    Generate the final response by passing the prompt and data to GPT.
    If no data is available from the database, GPT will answer directly.
    """
    logging.debug("Generating final response")
    messages = [
        {"role": "system",
         "content": "You are a conversational AI model named Iris born on 2024/10/01, that provides informative answers based on user prompts and available data."}
    ]

    # Add conversation history if needed
    if use_previous_response and is_follow_up:
        previous_response = response_history.get('previous_response', [])
        messages.append({
            "role": "assistant",
            "content": f"Previous Response: {json.dumps(previous_response, default=convert_to_serializable)}"
        })
        logging.debug("Included previous response in messages")

    if db_data:
        # Process database data and chunk it if necessary due to token limits
        db_data_serializable = convert_to_serializable(db_data)
        data_chunks = list(chunk_data(db_data_serializable))
        logging.debug("Data has been serialized and chunked")

        # Handle multi-chunk responses due to token constraints
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
                    max_tokens=5000,
                    temperature=0.7,  # Matching temperature from the second version
                    n=1,
                )
                # Clean the response before appending
                chunk_response = response.choices[0].message['content'].strip().replace('', '')
                final_response.append(chunk_response)
                logging.debug("Received response for chunk %d", idx + 1)

            except Exception as e:
                logging.error("Error generating GPT response: %s", e)
                return None

        logging.info("Successfully generated final response using data")
        return " ".join(final_response)

    else:
        # No new data; generate response based on previous context
        messages.append({
            "role": "user",
            "content": f"Based on the previous data, {prompt}"
        })
        logging.debug("No new data available; generating response based on previous context")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=5000,
                temperature=0.7,
                n=1,
            )
            logging.info("Successfully generated final response without new data")
            final_response = response.choices[0].message['content'].strip()

            # Clean the response
            final_response = final_response.replace('', '')  # Removing double asterisks
            return final_response

        except Exception as e:
            logging.error("Error generating GPT response: %s", e)
            return None


if __name__ == "__main__":
    app.run(debug=True)
