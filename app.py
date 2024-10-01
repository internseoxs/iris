from flask import Flask, request, jsonify, render_template
import openai
import psycopg2
import json
import sqlparse
import re
from decimal import Decimal
from datetime import datetime, date
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
response_history = {}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI and DB credentials
openai.api_key = os.getenv('OPENAI_API_KEY')

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')


def get_database_schema():
    """Retrieve table and column names from the PostgreSQL database."""
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

        return schema_dict

    except Exception as e:
        logging.error("Error fetching database schema: %s", e)
        return {}


def format_schema_for_gpt(schema):
    """Format the schema in a way that can be included in the prompt to GPT."""
    formatted_schema = ""
    for table, columns in schema.items():
        formatted_schema += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
    return formatted_schema


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/submit_query", methods=["POST"])
def submit_query():
    prompt = request.form.get('prompt')
    if not prompt:
        return render_template("contact.html", query=None, results=None, error="No prompt provided.")
    return render_template("about.html", prompt=prompt)


@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400

    prompt = data['prompt']
    schema = get_database_schema()
    if not schema:
        return jsonify({'error': 'Failed to retrieve database schema'}), 500

    formatted_schema = format_schema_for_gpt(schema)

    if requires_more_data(prompt):
        sql_query = generate_sql_query(prompt, formatted_schema)
        if not sql_query:
            return jsonify({'error': 'Failed to generate SQL query'}), 500

        logging.info("Generated SQL query: %s", sql_query)
        new_data = execute_sql_query(sql_query)
        if new_data is None:
            new_data = []

        response_history['previous_response'] = new_data
    else:
        new_data = response_history.get('previous_response', [])

    final_response = generate_final_response(prompt, new_data)
    if not final_response:
        return jsonify({'error': 'Failed to generate final response'}), 500

    return jsonify({'response': final_response})


@app.route("/reset", methods=["POST"])
def reset():
    global response_history
    response_history.clear()
    return jsonify({"message": "Response history cleared."}), 200


def requires_more_data(prompt):
    if not response_history.get('previous_response'):
        return True

    messages = [
        {"role": "system", "content": "Does the user's query need more data? Respond 'yes' or 'no'."},
        {"role": "user", "content": f"Prompt: {prompt}\nData: {json.dumps(response_history.get('previous_response', []), default=convert_to_serializable)}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=50,
            temperature=0.0,
            n=1,
        )
        return response.choices[0].message['content'].strip().lower() == 'yes'
    except Exception as e:
        logging.error("Error determining if more data is needed: %s", e)
        return True


def ensure_semicolon(sql_query):
    pattern = r"\bSELECT\b.*?(?=;|$)"
    match = re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL)
    if match:
        sql_query = match.group(0).strip()
        return sql_query + ';' if not sql_query.endswith(';') else sql_query
    return sql_query


def generate_sql_query(prompt, schema):
    messages = [
        {"role": "system", "content": f"Convert this to an SQL query: {schema}"},
        {"role": "user", "content": prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=2550,
            temperature=0,
            n=1,
        )
        sql_query = response.choices[0].message['content'].strip()
        return ensure_semicolon(sql_query)

    except Exception as e:
        logging.error("Error generating SQL query: %s", e)
        return None


def execute_sql_query(sql_query):
    if not is_safe_sql(sql_query):
        logging.warning("Unsafe SQL query: %s", sql_query)
        return []

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
        results = cursor.fetchmany(500)
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        return [dict(zip(colnames, row)) for row in results] if results else []
    except Exception as e:
        logging.error("Error executing SQL query: %s", e)
        return []


def is_safe_sql(query):
    """
    Simple check for SQL injection prevention by looking for dangerous keywords.
    This is a basic check and not a complete solution to SQL injection.
    """
    blacklist = ['DROP', 'DELETE', 'INSERT', 'UPDATE', '--', ';', '/*', '*/', '@@', '@', 'CHAR', 'NCHAR', 'VARCHAR', 'NVARCHAR', 'ALTER', 'BEGIN', 'CAST', 'CREATE', 'CURSOR', 'DECLARE', 'EXEC', 'FETCH', 'KILL', 'OPEN', 'SYSOBJECTS', 'SYS', 'XP_']
    query_upper = query.upper()

    for keyword in blacklist:
        if keyword in query_upper:
            logging.warning("Unsafe SQL detected: %s", keyword)
            return False
    return True


def convert_to_serializable(obj):
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj


def generate_final_response(prompt, db_data):
    final_response = []
    if db_data:
        db_data_serializable = convert_to_serializable(db_data)

        for chunk in chunk_data(db_data_serializable):
            messages = [
                {"role": "system", "content": "Analyze the following data."},
                {"role": "user", "content": f"Prompt: {prompt}\nData: {json.dumps(chunk)}"}
            ]

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=5000,
                    temperature=0.5,
                    n=1,
                )
                final_response.append(response.choices[0].message['content'].strip())

            except Exception as e:
                logging.error("Error generating GPT response: %s", e)
                return None
    else:
        messages = [
            {"role": "system", "content": "Answer the user's query."},
            {"role": "user", "content": f"Prompt: {prompt}\nNo data available."}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=5000,
                temperature=0.5,
                n=1,
            )
            final_response.append(response.choices[0].message['content'].strip())

        except Exception as e:
            logging.error("Error generating fallback response: %s", e)
            return None

    return " ".join(final_response)
