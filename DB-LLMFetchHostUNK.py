#!/usr/bin/env python3
"""
LLM Fetch Unknown Hosts

Fetches and updates unknown hosts for viruses using a local LLM with web search.
Uses LM Studio with DuckDuckGo search plugin to find actual hosts.

Usage:
    python DB-LLMFetchHostUNK.py

Requirements:
    - LM Studio server running on port 1234
    - DuckDuckGo search plugin enabled in LM Studio
    - db/db-viruses.sqlite3 with 'viruses' table

Input:
    - db/UNK_initial_host.txt (list of organisms with unknown hosts)
    - Or automatically extracts from database if file doesn't exist

Output:
    - db/UNK_LLM_hosts.tsv (TSV with Organism and Host columns)

Configuration:
    - LOCAL_SERVER_PORT: LM Studio server port (default: 1234)
    - Random delay: 2-4 seconds between requests

Model:
    - google/gemma-4-e4b (gemma-4-E4B-it-Q4_K_M.gguf)
    - Maximum tokens: content parameter

Notes:
    - Two-turn tool call loop: search -> final answer
    - Logs to logs/Process_DB-LLMFetchHostUNK_<timestamp>.log
"""

import os
import sys
import csv
import json
import time
import random
import re
import sqlite3
from datetime import datetime
from openai import OpenAI
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from config import *

LOCAL_SERVER_PORT = "1234"
client = OpenAI(
    base_url=f"http://localhost:{LOCAL_SERVER_PORT}/v1",
    api_key="local-server"
)

def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except:
        return None

def query_plugin_model(organism_name):
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
                "You are a precise bioinformatics assistant. Your task is to identify the natural host of organisms. "
                "Use the web_search tool if necessary. "
                "IMPORTANT: First check if the organism can infect humans. If yes, return 'Homo sapiens'. "
                "If not, return the natural host name. Do not include extra explanations or parenthesis. "
                "Example: If the natural host is 'Abutilon species (plants)', return 'Abutilon species'. "
                "Example: If the organism can infect humans, return 'Homo sapiens'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Find the natural biological host for: {organism_name}. "
                f"First determine if this organism can infect humans. "
                f"If yes, return 'Homo sapiens'. If no, return the natural host name. "
                f'Return the result strictly in this JSON format: {{"Organism": "{organism_name}", "Host": "Name"}}'
            )
        }
    ]

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
            return {"Organism": organism_name, "Host": "Unknown"}

        parsed_json = extract_json(raw_content)
        if parsed_json:
            return parsed_json

        return {"Organism": organism_name, "Host": "Unknown"}

    except Exception as e:
        print(f"  [!] Error processing {organism_name}: {e}")
        return {"Organism": organism_name, "Host": "Unknown"}

def main():
    setup_logging("DB-LLMFetchHostUNK")

    # Step 0: Create db/UNK_initial_host.txt by extracting organisms with Unknown host
    print("Step 0 - Extracting organisms with Unknown host from database...")
    conn = sqlite3.connect(SQLITE_CORR_VIRUSES_FILE)
    query = "SELECT DISTINCT organism FROM viruses WHERE host = 'Unknown' OR host = '' OR host IS NULL"
    df = conn.execute(query).fetchall()
    organisms = [row[0] for row in df if row[0] and row[0].strip()]
    conn.close()
    print(f"Total UNK hosts found: {len(organisms)}")
    print("\nSaving UNK organisms to UNK_INITIAL_HOST_FILE...")
    with open(UNK_INITIAL_HOST_FILE, 'w', encoding='utf-8') as f:
        for org in organisms:
            f.write(org + '\n')

    # Step 1: Read organisms from UNK_initial_host.txt
    print("Step 1 - Reading organisms from UNK_INITIAL_HOST_FILE...")
    try:
        with open(UNK_INITIAL_HOST_FILE, 'r', encoding='utf-8') as f:
            organisms = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: UNK_INITIAL_HOST_FILE not found.")
        return

    print(f"Total organisms loaded: {len(organisms)}")

    print("\nStep 2 - Finding hosts using LLM...")
    print(f"Processing {len(organisms)} organisms...")

    output_filename = UNK_LLM_HOSTS_FILE
    with open(output_filename, 'w', newline='', encoding='utf-8') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        writer.writerow(['Organism', 'Host'])

        for i, organism in enumerate(organisms):
            result = query_plugin_model(organism)
            host = result.get("Host", "Unknown")

            progress = ((i + 1) / len(organisms)) * 100
            print(f"[{progress:.1f}%] [{i+1}/{len(organisms)}] Organism: {organism} | Host: {host}")
            writer.writerow([organism, host])

            time.sleep(random.uniform(2.0, 4.0))

    print(f"\nTask complete! Saved to '{output_filename}'.")

if __name__ == "__main__":
    main()