"""
VirSentAI: Viral Sentry AI – Intelligent Zoonotic Surveillance Platform
This script is responsible for the initial setup of the SQLite database.
It checks if the database file exists at the path specified in the config.
If the database does not exist, it creates it and sets up the necessary 
tables: 'predictions' and 'models'. If the database already exists, it 
skips the creation process. Finally, it prints the schema of all tables 
in the database to the console and logs all operations to a timestamped 
log file.
"""
import os
import sqlite3
import logging
import datetime

from config import *

# --- Logging setup ---
# Get the script name without the .py extension
script_name = os.path.splitext(os.path.basename(__file__))[0]
# Create a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# create the log folder if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Define the log file name, including the subfolder
log_file_name = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

# Configure the logger to output to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filename=log_file_name,
    filemode='w'
)

# Path to the SQLite database
db_path = SQLite_FILE

db_exists = os.path.exists(db_path)

# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

if not db_exists:
    msg = f"Database not found. Creating a new one at: {db_path}"
    print(msg)
    logging.info(msg)
    # SQL query to create the predictions table
    create_predictions_table_query = PREDICTION_TABLE_QUERY
    cursor.execute(create_predictions_table_query)

    # SQL query to create the models table
    create_models_table_query = MODELS_TABLE_QUERY
    cursor.execute(create_models_table_query)
    
    conn.commit()
    msg = "Database and tables created successfully."
    print(msg)
    logging.info(msg)
else:
    msg = f"Database already exists at: {db_path}"
    print(msg)
    logging.info(msg)

msg = "\nDatabase schema:"
print(msg)
logging.info(msg)
# Get all table names from the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

if not tables:
    msg = "The database is empty (no tables found)."
    print(msg)
    logging.info(msg)
else:
    # For each table, get and print its schema
    for table_name in tables:
        table_name = table_name[0]
        msg = f"\n-- Table: {table_name}"
        print(msg)
        logging.info(msg)
        msg = "   Fields:"
        print(msg)
        logging.info(msg)
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for column in columns:
            # column format: (id, name, type, notnull, default_value, pk)
            msg = f"     - {column[1]} ({column[2]})"
            print(msg)
            logging.info(msg)

# Close the connection
conn.close()
