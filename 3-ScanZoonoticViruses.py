"""
VirSentAI: Viral Sentry AI – Intelligent Zoonotic Surveillance Platform
This script scans the NCBI nucleotide database for new virus sequences within a 
specified date range: only complete viral genomes from non-human hosts, 
fetches Virus ID, Name, Sequence, Host, and Registration Date as TSV
You can set flags for the following steps:
- Scan for new viruses from NCBI
- Predict with a fine-tuned HyenaDNA model
- Add new predictions to an SQLite database
- Create TSV files with all predictions and models in db folder
- Create new JSON data for the webapp
- Update HTML page with the latest scan date
"""
import os
import time
from datetime import datetime, timedelta
from Bio import Entrez
import csv
import logging
import pandas as pd
import sqlite3
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from scipy.special import softmax
import json
import re
import shutil
import zipfile
from config import *

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------

# Flags to control script behavior
f_scan       = True  # If True, scan NCBI for new viruses
f_predict    = True  # If True, use the model to predict on new viruses
f_add2db     = True  # If True, add new predictions to the SQLite database
f_webapp     = True  # If True, create new JSON data for the webapp and update HTML page

# Define the date range for the search
start_date = "2025/10/24"
end_date   = "2025/10/25"

# Reformat dates from YYYY/MM/DD to YYYY-MM-DD
start_date_formatted = datetime.strptime(start_date, "%Y/%m/%d").strftime("%Y-%m-%d")
end_date_formatted   = datetime.strptime(end_date,   "%Y/%m/%d").strftime("%Y-%m-%d")

# Output file and folder
output_folder = VIRUS_SCAN_DIR
output_file = os.path.join(output_folder, f"virus_scan_{start_date_formatted}_to_{end_date_formatted}.tsv")
output_prediction_path = os.path.splitext(output_file)[0] + "_predictions.tsv"
# ---------------------------------------------

# Create Virus scan folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Logging setup ---
# Get the script name without the .py extension
script_name = os.path.splitext(os.path.basename(__file__))[0]
# Create a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# create the log folder if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Define the log file name, including the subfolder
log_file_name = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler()
    ]
)

# Function to fetch virus data from NCBI
def fetch_virus_data(start_date, end_date):
    viruses = []
    try:
        # Search query for complete virus genomes with non-human hosts between dates
        # search_term = (
        #     "viruses[Organism] NOT 9606[TaxID] "
        #     "AND (\"complete genome\"[All Fields] OR \"complete sequence\"[All Fields]) "
        #     f"AND ({start_date}[PDAT] : {end_date}[PDAT])"
        # )

        search_term = (
            "txid10239[Organism:exp] AND complete genome[Title] NOT human[Host]"
            f"AND ({start_date}[PDAT] : {end_date}[PDAT])"
        )

        msg = f"Search term: {search_term}"
        logging.info(msg)
        Entrez.email = NCBI_ENTREZ_EMAIL  # Replace with your email for NCBI API usage
        handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=NCBI_ENTREZ_MAX_RESULTS)
        record = Entrez.read(handle)
        handle.close()

        id_list = record.get("IdList", [])
        msg = f"Number of hits: {len(id_list)}"
        logging.info(msg)

        # stop the function if the maximum results is NCBI_ENTREZ_MAX_RESULTS
        if len(id_list) >= NCBI_ENTREZ_MAX_RESULTS:
            msg = f"Warning: Number of hits reached the maximum limit of {NCBI_ENTREZ_MAX_RESULTS}. Consider increasing the limit or narrowing the date range."
            logging.warning(msg)
            exit()

        if not id_list:
            msg = "No new virus sequences found matching the criteria. The script will exit."
            logging.info(msg)
            exit()

        # Fetch details for each ID
        no = 1
        for virus_id in id_list:
            logging.info(f"-> Processing {no}/{len(id_list)}: NCBI ID {virus_id}")
            no += 1
            handle = Entrez.efetch(db="nucleotide", id=virus_id, rettype="gb", retmode="text")
            gb_record = handle.read()
            handle.close()

            # Parse the GenBank record
            lines = gb_record.split("\n")
            virus_name = ""
            genbank_id = ""
            sequence = ""
            host = ""
            registration_date = ""
            in_sequence = False

            for line in lines:
                # Extract GenBank ID
                if line.startswith("VERSION"):
                    parts = line.split()
                    if len(parts) > 1:
                        genbank_id = parts[1].strip()

                # Extract virus name
                elif line.startswith("DEFINITION"):
                    virus_name = line.split("DEFINITION")[1].strip()

                # Extract registration date from LOCUS
                elif line.startswith("LOCUS") and not registration_date:
                    # Example: LOCUS       NC_045512      29903 bp    RNA     linear   VRL 13-APR-2020
                    locus_date = line.split()[-1].strip()  # Last field: DD-MMM-YYYY
                    try:
                        registration_date = datetime.strptime(locus_date, "%d-%b-%Y").strftime("%Y-%m-%d")
                    except ValueError:
                        registration_date = locus_date # If the date format is unexpected, keep the original value

                # Extract host information
                elif "/host=" in line:
                    host = line.split("/host=")[1].replace("\"", "").strip()

                # Extract sequence
                elif line.startswith("ORIGIN"):
                    in_sequence = True
                    continue
                elif in_sequence and line.strip() and not line.startswith("//"):
                    sequence += "".join(line.split()[1:]).replace(" ", "").replace("\n", "")
                elif line.startswith("//"):
                    in_sequence = False

            # Secondary filter: Check if the host is human and skip if so
            if "homo sapiens" in host.lower():
                msg = f"Skipping record {genbank_id} because host is 'Homo sapiens'."
                logging.info(msg)
                continue

            if "homo sapien" in host.lower():
                msg = f"Skipping record {genbank_id} because host is 'Homo sapien'."
                logging.info(msg)
                continue

            if "covid-19" in host.lower():
                msg = f"Skipping record {genbank_id} because host is 'COVID-19'."
                logging.info(msg)
                continue

            # if not human host
            if virus_name and sequence and genbank_id and registration_date:
                viruses.append({
                    "Virus_ID": genbank_id,
                    "Virus_Name": virus_name,
                    "Virus_Seq": sequence.upper(),
                    "Host": host if host else "Unknown",
                    "Registration_Date": registration_date,
                    "Database": "NCBI"
                })

                msg = f"-> Processed Virus_ID (GenBank): {genbank_id}, Registration Date: {registration_date}"
                logging.info(msg)
                time.sleep(0.2)  # Respect NCBI usage guidelines
            else:
                msg = f"Skipping record with NCBI ID {virus_id}: missing GenBank ID, name, sequence, or registration date."
                logging.info(msg)

        return viruses

    except Exception as e:
        msg = f"Error fetching data: {e}"
        logging.error(msg)
        print(msg)
        return viruses

def predict_with_model(model_dir, input_tsv_path, output_tsv_path):
    """
    Uses a fine-tuned HyenaDNA model to make predictions on new virus sequences from a TSV file.

    Args:
        model_dir (str): The directory of the fine-tuned model.
        input_tsv_path (str): The path to the input TSV file from the virus scan.
        output_tsv_path (str): The path for the output TSV file with predictions.
    """
    try:
        msg = "\nStarting prediction process..."
        logging.info(msg)

        # --- Device setup ---
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")
        msg = f"Using device: {device}"
        logging.info(msg)

        # --- Load model and tokenizer ---
        msg = f"Loading fine-tuned model and tokenizer from {model_dir}..."
        logging.info(msg)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # --- Load data ---
        msg = f"Loading new data from {input_tsv_path}..."
        logging.info(msg)
        df_new = pd.read_csv(input_tsv_path, sep='\t')

        # --- Tokenization ---
        msg = "Starting tokenization process..."
        logging.info(msg)
        prediction_dataset = Dataset.from_pandas(df_new[['Virus_Seq']].rename(columns={"Virus_Seq": "seq"}))

        def tokenize_function(examples):
            return tokenizer(
                examples["seq"],
                truncation=True,
                padding="max_length",
                max_length=SEQ_MAX_LENGTH,
            )

        tokenized_prediction_dataset = prediction_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["seq"]
        )
        tokenized_prediction_dataset.set_format("torch")

        # --- Perform Predictions ---
        training_args = TrainingArguments(
            output_dir=os.path.join(model_dir, PREDICT_TEMP_SUBFOLDER),
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            bf16=True,
            report_to="none",
        )
        trainer = Trainer(model=model, args=training_args)

        torch.cuda.empty_cache()
        msg = "Starting predictions on the new data..."
        logging.info(msg)
        predictions = trainer.predict(tokenized_prediction_dataset)
        
        # --- Process and Save Predictions ---
        # The output of trainer.predict is a PredictionOutput object which contains
        # predictions (logits), label_ids (if available), and metrics.
        logits = predictions.predictions

        # Apply softmax to the logits to get probabilities
        # The logits are likely in bfloat16, so convert to float32 for stable softmax computation.
        probabilities = softmax(torch.from_numpy(logits).float().numpy(), axis=1)

        # The probability of the positive class (Class 1) is in the second column.
        prob_class_1 = probabilities[:, 1]

        # Add the probabilities as a new column to the original dataframe
        df_new['PClass_1'] = prob_class_1

        msg = f"Saving predictions to {output_tsv_path}..."
        logging.info(msg)
        df_new.to_csv(output_tsv_path, sep='\t', index=False)
        msg = f"Predictions saved successfully."
        logging.info(msg)

    except FileNotFoundError as e:
        error_msg = f"Error: File not found - {e}"
        print(error_msg)
        logging.error(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        logging.error(error_msg)

def add_scan_data_to_db(tsv_file_path, sqlite_db_path, modelid):
    """
    Reads virus scan data from a TSV file and inserts it into the 'predictions' table in the SQLite database.

    Args:
        sqlite_db_path (str): The full path to the SQLite database.
        tsv_file_path (str): The full path to the TSV file containing scan data.
    """

    msg = f"Reading scan data from: {tsv_file_path}"
    logging.info(msg)

    try:
        # Read the TSV file into a pandas DataFrame
        df = pd.read_csv(tsv_file_path, sep='\t')
        msg = f"Found {len(df)} new viruses to process."
        logging.info(msg)

        # Connect to the SQLite database
        msg = f"Connecting to database: {sqlite_db_path}"
        logging.info(msg)
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        # SQL query to insert a prediction
        insert_query = """
        INSERT INTO predictions (
            prediction_date,
            prediction_score,
            model_id,
            virus_id,
            virus_name,
            virus_host,
            virus_db,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        # Iterate over the DataFrame rows and insert into the database
        # Virus_ID	Virus_Name	Virus_Seq	Host	Registration_Date	Database	PClass_1
        for index, row in df.iterrows():
            prediction_data = (
                row['Registration_Date'],
                row['PClass_1'],
                modelid,
                row['Virus_ID'],
                row['Virus_Name'],
                row['Host'],
                row['Database'],
                datetime.now()
            )
            cursor.execute(insert_query, prediction_data)

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        msg = f"\nSuccessfully inserted {len(df)} new virus records into the database."
        print(msg)
        logging.info(msg)

    except FileNotFoundError:
        msg = f"Error: The file was not found at {tsv_file_path}"
        print(msg)
        logging.error(msg)
    except Exception as e:
        msg = f"An error occurred: {e}"
        print(msg)
        logging.error(msg)

def embed_json_into_html(html_template_path, json_data_path, output_html_path):
    """
    Reads a JSON file, embeds its content into a specified HTML file,
    and saves the result to a new HTML file.

    Args:
        html_template_path (str): The path to the input HTML file (e.g., "index.html").
        json_data_path (str): The path to the JSON data file (e.g., "summary_stats.json").
        output_html_path (str): The path for the output HTML file (e.g., "index_updated.html").
    """
    try:
        # 1. Read the JSON data
        with open(json_data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Convert the Python dictionary to a JSON string, formatted for JavaScript
        # Using indent=10 to match the original HTML's indentation for readability
        json_string_for_js = json.dumps(json_data, indent=10)

        # Construct the new JavaScript block for summaryStats
        # It includes the original comment and 'const summaryStats =' declaration
        new_summary_stats_js_block = f"""      // Data from summary_stats.json embedded directly
      const summaryStats = {json_string_for_js};"""

        # 2. Read the HTML template
        with open(html_template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

       # 3. Find and replace the OLD 'const summaryStats = {...};' block
        # The pattern looks for the entire old block:
        # \s*// Data from summary_stats\.json embedded directly
        # Followed by non-greedy characters (.*?) until the closing object and semicolon: \{.*?};\s*
        pattern = re.compile(
            r'\s*// Data from summary_stats\.json embedded directly.*?const summaryStats\s*=\s*\{.*?};\s*',
            re.DOTALL
        )
        
        # Check if the pattern is found
        match = pattern.search(html_content)
        if match:
            # Replace the matched block with the new_summary_stats_js_block
            updated_html_content = pattern.sub(new_summary_stats_js_block, html_content, 1)
            print(f"Successfully replaced 'const summaryStats' block in '{html_template_path}'.")
        else:
            print(f"Warning: The 'const summaryStats = {{...}};' block was not found in '{html_template_path}'.")
            print("No replacement was made. Please ensure the HTML file contains this structure.")
            updated_html_content = html_content # Keep original content if pattern not found

        # 4. Write the updated HTML content to a new file (index_updated.html)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(updated_html_content)

        print(f"Successfully created '{output_html_path}' with embedded JSON data.")

    except FileNotFoundError as e:
        print(f"Error: One of the specified files was not found: {e}")
        # Re-raise the exception to stop further execution if a file is missing
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{json_data_path}': {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during embedding: {e}")
        raise


# Main execution
if __name__ == "__main__":

    # --- Scan for new viruses ---
    if f_scan == True:
        msg = f"\nFetching virus data from NCBI between {start_date} and {end_date}..."
        logging.info(msg)

        if os.path.exists(output_file):
            msg = f"Output file already exists: {output_file}. Scan will be skipped."
            logging.info(msg)
        else:
            msg = f"Scanning for virus sequences from {start_date} to {end_date}..."
            logging.info(msg)
            virus_data = fetch_virus_data(start_date, end_date)

            if virus_data:
                with open(output_file, "w", newline="", encoding="utf-8") as tsvfile:
                    writer = csv.DictWriter(tsvfile, fieldnames=["Virus_ID", "Virus_Name", "Virus_Seq", "Host", "Registration_Date", "Database", ], delimiter="	")
                    writer.writeheader()
                    writer.writerows(virus_data)
                msg = f"Data saved to {output_file} with {len(virus_data)} records."
                logging.info(msg)
            else:
                msg = "No data to save."
                logging.info(msg)

        # --- Verification Step ---
        if os.path.exists(output_file):
            msg = "Verifying output file ---"
            logging.info(msg)
            msg = f"Reading the first 2 data rows from {output_file}..."
            logging.info(msg)
            try:
                with open(output_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    
                    # Read and log header
                    header = next(reader)
                    msg = " | ".join(header)
                    logging.info(msg)
                    
                    separator = "-" * len(msg)
                    logging.info(separator)

                    # Read and log the first two data rows
                    for i in range(2):
                        try:
                            row = next(reader)
                            # To avoid excessively long lines in the log/console, we can truncate the sequence
                            row_to_log = row[:]
                            if len(row_to_log) > 2 and len(row_to_log[2]) > 50:
                                    row_to_log[2] = row_to_log[2][:50] + "..."
                            
                            msg = " | ".join(row_to_log)
                            logging.info(msg)
                        except StopIteration:
                            # Less than 2 rows of data in the file
                            break
            except Exception as e:
                msg = f"An error occurred during verification: {e}"
                logging.error(msg)
                print(msg)
        # end verification
    
        # (If needed, a second pass was previously present — removed duplicate sequence.)
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')

                # Read and log header
                header = next(reader)
                msg = " | ".join(header)
                logging.info(msg)

                separator = "-" * len(msg)
                logging.info(separator)

                # Read and log the first two data rows
                for i in range(2):
                    try:
                        row = next(reader)
                        # To avoid excessively long lines in the log/console, we can truncate the sequence
                        row_to_log = row[:]
                        if len(row_to_log) > 2 and len(row_to_log[2]) > 50:
                            row_to_log[2] = row_to_log[2][:50] + "..."
                        
                        msg = " | ".join(row_to_log)
                        logging.info(msg)
                    except StopIteration:
                        # Less than 2 rows of data in the file
                        break
        except Exception as e:
            msg = f"An error occurred during verification: {e}"
            logging.error(msg)
            print(msg)
    else:
        msg = "! Virus scan step skipped as per configuration."
        logging.info(msg)
    
    # --- Predict with model ---
    if f_predict == True:
        
        if os.path.exists(output_prediction_path):
            msg = f"Prediction file already exists: {output_prediction_path}. Prediction will be skipped."
            logging.info(msg)
        else:
            predict_with_model(NEW_MODEL_DIR, output_file, output_prediction_path)
    else:
        msg = "! Prediction step skipped as per configuration."
        logging.info(msg)

    # --- Add predictions for new viruses to SQLite db ---
    if f_add2db == True:
        # Example of how to call the new function
        add_scan_data_to_db(output_prediction_path, SQLite_FILE, modelid=2)  # Assuming model ID 2 for this example

        # --- Verification of data insertion ---
        msg = "\n--- Verifying data insertion into database ---"
        logging.info(msg)
        
        try:
            db_path = SQLite_FILE
            conn = sqlite3.connect(db_path)
            
            msg = f"\nChecking the last 5 entries in the 'predictions' table of {db_path}..."
            logging.info(msg)

            # Query to select the last 5 predictions for verification
            verify_query = "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT 5;"
            df_verify = pd.read_sql_query(verify_query, conn)
            
            if not df_verify.empty:
                msg = "Verification successful. Last 5 predictions:"
                logging.info(msg)
                logging.info(df_verify.to_string())
            else:
                msg = "Verification check: No predictions found in the table."
                logging.info(msg)

            # Close the connection
            conn.close()

        except sqlite3.Error as e:
            msg = f"Database error during verification: {e}"
            logging.error(msg)
        except Exception as e:
            msg = f"An unexpected error occurred during verification: {e}"
            logging.error(msg)
    else:
        msg = "! Add to SQLite database step skipped as per configuration."
        logging.info(msg)
    
    # --- Create new JSON and TSV files and update HTML page ---
    # (insert JSON data into the HTML page)
    if f_webapp == True:
        # --- Load data from SQLite database ---
        try:
            msg = "Loading data from SQLite database for webapp update..."
            logging.info(msg)
            conn = sqlite3.connect(SQLite_FILE)
            df_db  = pd.read_sql_query("SELECT * FROM predictions", conn) # predictions table
            df_db2 = pd.read_sql_query("SELECT * FROM models", conn) # models table
            conn.close()
            logging.info(f"Successfully loaded {len(df_db)} records from {SQLite_FILE}")

            # --- Create TSV file with all the predictions ---
            msg = f"Saving all predictions to TSV file: {ALL_PREDS_FILE}"
            logging.info(msg)
            # Save the DataFrame to a TSV file
            # Drop the 'created_at' column
            df_db = df_db.drop('created_at', axis=1)
            # Sort by prediction_score descending
            df_db.sort_values(by='prediction_score', ascending=False, inplace=True)
            df_db.to_csv(ALL_PREDS_FILE, sep='\t', index=False)

            # --- Create TSV file with all the models ---
            msg = f"Saving all models to TSV file: {ALL_MODELS_FILE}"
            logging.info(msg)
            # Save the DataFrame to a TSV file
            # Drop the 'created_at' column
            df_db2 = df_db2.drop('created_at', axis=1)
            df_db2.to_csv(ALL_MODELS_FILE, sep='\t', index=False)

            # --- Create JSON Summary from Database Data ---
            # Ensure prediction_date is in datetime format
            df_db['prediction_date'] = pd.to_datetime(df_db['prediction_date'])

            # Calculate total scanned viruses
            total_scanned_db = len(df_db)

            # Calculate total zoonotic viruses (prediction_score >= 0.8)
            zoonotic_df_db = df_db[df_db['prediction_score'] >= AUROC_CUTOFF].copy()
            total_zoonotic_db = len(zoonotic_df_db)

            # Calculate monthly new cases (for prediction_score >= 0.8)
            zoonotic_df_db['Month'] = zoonotic_df_db['prediction_date'].dt.to_period('M').astype(str)
            monthly_new_cases_db = zoonotic_df_db.groupby('Month').size().to_dict()

            # Top 10 predictions (prediction_score >= 0.8, ordered by score)
            top_preds_db = (
                zoonotic_df_db
                .sort_values(by='prediction_score', ascending=False)
                .head(10)
                .loc[:, ['virus_id', 'virus_name', 'virus_host', 'virus_db', 'prediction_score', 'prediction_date']]
                .rename(columns={
                    'virus_id': 'Virus ID',
                    'virus_name': 'Virus Name',
                    'virus_host': 'Virus Host',
                    'virus_db': 'Database',
                    'prediction_score': 'Probability',
                    'prediction_date': 'Scan Date'
                })
            )
            
            # Convert date to string for JSON serialization
            top_preds_db['Scan Date'] = top_preds_db['Scan Date'].dt.strftime('%Y-%m-%d')
            top_preds_db = top_preds_db.to_dict(orient='records')
            
            # Create summary dictionary
            summary_db = {
                "Total Scanned Viruses": total_scanned_db,
                "Zoonotic Viruses": total_zoonotic_db,
                "Monthly New Cases": monthly_new_cases_db,
                "Top Predictions": top_preds_db
            }

            # Save to JSON file
            with open(JSON_FILE, 'w') as f:
                json.dump(summary_db, f, indent=4)

            logging.info(f"Summary JSON from database saved to {JSON_FILE}")

            # --- Embed JSON data into HTML page ---
            # Define file paths
            html_template_file = os.path.join(WEBAPP_DIR, "index.html") # "index.html"
            json_input_file = JSON_FILE
            temp_updated_html_file = os.path.join(WEBAPP_DIR, "index_updated.html") # "index_updated.html" # Temporary file for the new HTML
            old_html_backup_file   = os.path.join(WEBAPP_DIR, "index_old.html") # "index_old.html"      # Backup of the current index.html

            try:
                # 0. Ensure summary_stats_from_db.json exists for the script to run meaningfully
                if not os.path.exists(json_input_file):
                    logging.info(f"Error: JSON data file '{json_input_file}' not found. Please create it or adjust the path.")
                    exit(1)
                
                # 1. Copy index.html to index_old.html (backup the current version)
                if os.path.exists(html_template_file):
                    shutil.copyfile(html_template_file, old_html_backup_file)
                    logging.info(f"Backed up current '{html_template_file}' to '{old_html_backup_file}'.")
                else:
                    logging.info(f"Warning: '{html_template_file}' not found. Cannot create a backup. Proceeding with generation.")

                # 2. Call the function to create the updated HTML content in a temporary file
                embed_json_into_html(html_template_file, json_input_file, temp_updated_html_file)

                # 3. Copy index_updated.html to index.html (replace the main HTML file)
                shutil.copyfile(temp_updated_html_file, html_template_file)
                logging.info(f"Copied '{temp_updated_html_file}' to '{html_template_file}'.")

                # 4. Remove index_updated.html (clean up the temporary file)
                os.remove(temp_updated_html_file)
                logging.info(f"Removed temporary file '{temp_updated_html_file}'.")

                logging.info("\nScript completed successfully. Your 'index.html' has been updated, and 'index_old.html' contains the previous version.")

            except Exception as e:
                logging.info(f"\nAn error occurred during the script execution: {e}")
                logging.info("Please check the error messages above for details.")


        except FileNotFoundError:
            logging.info(f"Error: The database file was not found at {SQLite_FILE}")
        except Exception as e:
            logging.info(f"An error occurred: {e}")
    else:
        msg = "! Webapp update step skipped as per configuration."
        logging.info(msg)
