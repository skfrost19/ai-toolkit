import os
import requests
from datetime import datetime
import time
import logging
from dotenv import load_dotenv

load_dotenv()
# logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:  %(message)s")


def supabase_upsert(system_prompt, body, hyde, all_examples, full_text):
    try:
        dt = datetime.now()
        dt_str = dt.isoformat()

        user_id = body["user_name"]
        template_category = body["template_category"]
        template_language = body["template_language"]
        message_input = body["message_input"]
        optimization_choice = body["optimization_choice"]
        contact = body["contact"]
        email_id = body["email_id"]
        mood = body["mood"]
        response1 = full_text
        hypothetical_message = hyde
        system_prompt = system_prompt

        data_to_save = {
            "user_id": user_id,
            "name": user_id,
            "contact": contact,
            "email_id": email_id,
            "mood": mood,
            "template_language": template_language,
            "template_category": template_category,
            "message_input": message_input,
            "optimization_choice": optimization_choice,
            "hypothetical_message": hypothetical_message,
            "response1": response1,
            "all_examples": all_examples,
            "timestamp": dt_str,
            "system_prompt": system_prompt,
        }
        table_name = os.environ["SUPABASE_TABLE_NAME"]

        start_time = time.time()
        status_code, res = insert_table_data(table_name, data_to_save)
        end_time = time.time()
        print("Supabase Insert Latency: ", end_time - start_time)
        logging.info("Supabase Insert Latency: " + str(end_time - start_time))

        if status_code == 201:
            print("Data saved successfully")
        else:
            print("Error in saving data: ", res.text)

    except Exception as e:
        logging.error(f"Error in supabase_upsert: {e}")
        raise e


def insert_table_data(table_name, data):
    try:
        SUPABASE_URL = os.environ["SUPABASE_URL"]
        SUPABASE_API_KEY = os.environ["SUPABASE_API_KEY"]
    except Exception as e:
        logging.error(f"Error in getting environment variables: {e}")
        raise e
    try:
        base_url = f"{SUPABASE_URL}/rest/v1"

        # Set the headers for the API requests
        headers = {
            "apikey": SUPABASE_API_KEY,
            "Authorization": f"Bearer {SUPABASE_API_KEY}",
            "Content-Type": "application/json",
        }
        url = f"{base_url}/{table_name}"
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 201:
            logging.error(f"Failed to insert data into Supabase: {response}")
            raise
        return response.status_code, response
    except Exception as e:
        logging.error(f"Error in insert_table_data: {e}")
        raise e
