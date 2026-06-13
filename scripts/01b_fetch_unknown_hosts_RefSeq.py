#!/usr/bin/env python3
"""
Step 1b: Fetch Unknown Hosts Using LLM

Reads raw RefSeq TSV from Step 1a, finds hosts for viruses where host is "Unknown"
using gemma4 LLM with DuckDuckGo search.

Usage:
    python 01b_fetch_unknown_hosts_RefSeq.py --start-date 2026/04/16 --end-date 2026/05/04

Input:
    - virus-scan/virus_scan_YYYY-MM-DD_to_YYYY-MM-DD_RefSeq_raw.tsv (from Step 1a)
    - --start-date: Start date in YYYY/MM/DD format (required)
    - --end-date: End date in YYYY/MM/DD format (required)

Output:
    - virus-scan/virus_scan_2026-04-16_to_2026-05-04_RefSeq.tsv

Process:
    1. Read raw TSV from Step 1a
    2. Identify records where Host = "Unknown"
    3. For each unknown host, query gemma4 with DuckDuckGo search
    4. Add LLM_host column:
       - "1" if host was modified by gemma4
       - "0" if host was from RefSeq or remained Unknown
    5. Filter out human hosts (Homo sapiens)

Requirements:
    - LM Studio server running on port 1234
    - DuckDuckGo search plugin enabled
    - Model: google/gemma-4-e4b (gemma-4-E4B-it-Q4_K_M.gguf)

Notes:
    - Maximum tokens content parameter
    - Logs to logs/01b_fetch_unknown_hosts_RefSeq_<timestamp>.log
    - If the output TSV file already exists, the process will be skipped. Delete the file to re-run.

Requirements:
    - LM Studio server running on port 1234 (the model is not specified inside the script)
    - DuckDuckGo search plugin enabled in LM Studio
    - Raw TSV file from 01a_scan_viruses_raw_RefSeq.py must exist
"""
import os
import csv
import logging
import json
import re
import time
import argparse
from datetime import datetime
import pandas as pd
from openai import OpenAI

from config import *

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
parser = argparse.ArgumentParser(description="Fetch unknown hosts using gemma4 (RefSeq)")
parser.add_argument("--start-date", required=True, help="Start date in YYYY/MM/DD format")
parser.add_argument("--end-date", required=True, help="End date in YYYY/MM/DD format")
args = parser.parse_args()

# Get dates from command line arguments
start_date = args.start_date
end_date   = args.end_date

# Reformat dates from YYYY/MM/DD to YYYY-MM-DD
start_date_formatted = datetime.strptime(start_date, "%Y/%m/%d").strftime("%Y-%m-%d")
end_date_formatted   = datetime.strptime(end_date,   "%Y/%m/%d").strftime("%Y-%m-%d")

# Input and output files
output_folder = VIRUS_SCAN_DIR
input_file = os.path.join(output_folder, f"virus_scan_{start_date_formatted}_to_{end_date_formatted}_RefSeq_raw.tsv")
output_file = os.path.join(output_folder, f"virus_scan_{start_date_formatted}_to_{end_date_formatted}_RefSeq.tsv")

print(f"Processing unknown hosts (RefSeq) from {start_date} to {end_date}")

# Create Virus scan folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Logging setup ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_name = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler()
    ]
)

def query_gemma_for_host(virus_name):
    """
    Uses local gemma4 model with DuckDuckGo search to extract host from Virus_Name.
    Checks if the host can infect humans - if yes, marks as human to be filtered out.
    """
    LOCAL_SERVER_PORT = "1234"
    client = OpenAI(
        base_url=f"http://localhost:{LOCAL_SERVER_PORT}/v1",
        api_key="local-server"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "duckduckgo_search",
                "description": "Search the web using DuckDuckGo to find up-to-date biological information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (e.g., 'host of Influenza A virus')"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise bioinformatics assistant. Your task is to identify the natural host of viruses. "
                "Use the web_search tool if necessary. "
                "IMPORTANT: First check if the virus can infect humans. If yes, return 'Homo sapiens'. "
                "If not, return the natural host name. Do not include extra explanations or parenthesis. "
                "Example: If the natural host is 'birds', return 'birds'. "
                "Example: If the virus can infect humans, return 'Homo sapiens'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Find the natural biological host for this virus based on: {virus_name}. "
                f"First determine if this virus can infect humans. "
                f"If yes, return 'Homo sapiens'. If no, return the natural host name. "
                f'Return the result strictly in this JSON format: {{"host_llm": "Host_Name"}}'
            )
        }
    ]

    def extract_json(text):
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text)
        except:
            return None

    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            seed=42,
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            messages.append(response_message)

            plugin_output = getattr(tool_call.function, "output", "") or ""

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": plugin_output
            })

            follow_up = client.chat.completions.create(
                model="local-model",
                messages=messages,
                tools=tools,
                tool_choice="none",
                temperature=0.0,
                seed=42,
            )
            raw_content = follow_up.choices[0].message.content or ""
        else:
            raw_content = response_message.content or ""

        if not raw_content:
            return "Unknown"

        parsed_json = extract_json(raw_content)
        if parsed_json:
            return parsed_json.get('host_llm', 'Unknown')

        return "Unknown"

    except Exception as e:
        return f"LLM API Error: {e}"

def process_unknown_hosts():
    """
    Reads the raw RefSeq TSV, finds hosts for Unknown entries using gemma4,
    and saves the corrected TSV.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        print(f"Error: Input file {input_file} not found. Run 01a_scan_viruses_raw_RefSeq.py first.")
        return []

    # Read the raw TSV
    logging.info(f"Reading raw RefSeq TSV file: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    logging.info(f"Loaded {len(df)} records from raw RefSeq TSV")

    # Initialize LLM_host column to 0 (not modified by LLM)
    df['LLM_host'] = '0'

    # Find records with Unknown host
    unknown_hosts = df[df['Host'] == 'Unknown']
    logging.info(f"Found {len(unknown_hosts)} records with Unknown host")

    # Process each unknown host
    if len(unknown_hosts) > 0:
        for idx, row in unknown_hosts.iterrows():
            virus_id = row['Virus_ID']
            virus_name = row['Virus_Name']
            
            logging.info(f"Processing unknown host for {virus_id}: {virus_name[:50]}...")
            
            # Query gemma4 for host
            llm_host_result = query_gemma_for_host(virus_name)
            
            if "LLM API Error" in llm_host_result:
                logging.warning(f"   -> LLM failed for ID {virus_id}: {llm_host_result}")
                df.loc[idx, 'Host'] = "Unknown"
                df.loc[idx, 'LLM_host'] = "0"
            else:
                # Update host and mark as LLM-modified
                df.loc[idx, 'Host'] = llm_host_result
                df.loc[idx, 'LLM_host'] = "1"
                logging.info(f"   -> LLM found host: {llm_host_result}")
            
            # Rate limiting
            time.sleep(1.0)

    # Filter out human hosts
    logging.info("Filtering out human hosts...")
    initial_count = len(df)
    df = df[~df['Host'].str.lower().str.contains('homo sapiens', na=False)]
    df = df[~df['Host'].str.lower().str.contains('covid-19', na=False)]
    filtered_count = initial_count - len(df)
    logging.info(f"Filtered out {filtered_count} human host records. Remaining: {len(df)}")

    # Save the corrected TSV
    logging.info(f"Saving corrected RefSeq TSV to: {output_file}")
    df.to_csv(output_file, sep='\t', index=False)
    logging.info(f"Saved {len(df)} records to {output_file}")

    return df

if __name__ == "__main__":
    print(f"Processing unknown hosts (RefSeq) from {start_date} to {end_date}...")
    
    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}. Process will be skipped.")
    else:
        result_df = process_unknown_hosts()
        
        if not result_df.empty:
            print(f"Data saved to {output_file} with {len(result_df)} records.")
        else:
            print("No data to save.")

    # Verification step
    if os.path.exists(output_file):
        print("\n--- Verification ---")
        print(f"Reading first 2 rows from {output_file}...")
        try:
            df = pd.read_csv(output_file, sep='\t', nrows=2)
            print(df.to_string())
        except Exception as e:
            print(f"Error during verification: {e}")