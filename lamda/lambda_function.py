import json
import requests
import logging
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Notion API credentials
notion_token = "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"  # Replace with your Notion integration token
notion_database_id = "12f63685a48180298ef2d10867d62bd6"  # Replace with your Notion database ID

def send_notion_notification(filename: str, action: str, status: str, elapsed_time: float):
    """Send a notification to Notion with the status of the Lambda call to FastAPI."""
    message = f"The Lambda function for '{filename}' {action} request {status} in {elapsed_time:.2f} seconds."
    notification_time = datetime.now().isoformat()
    
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    # Data payload for Notion API
    data = {
        "parent": {"database_id": notion_database_id},
        "properties": {
            "Title": {
                "title": [{"text": {"content": filename}}]
            },
            "Status": {
                "select": {"name": status}
            },
            "Action": {
                "rich_text": [{"text": {"content": action}}]
            },
            "Elapsed Time (s)": {
                "number": elapsed_time
            },
            "Timestamp": {
                "date": {"start": notification_time}
            },
            "Message": {
                "rich_text": [{"text": {"content": message}}]
            }
        }
    }
    
    response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=data)
    
    if response.status_code == 200:
        logger.info("Notion notification added successfully.")
    else:
        logger.error("Failed to send Notion notification: %s", response.text)

def lambda_handler(event, context):
    # Extract S3 bucket and object key from the event
    try:
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        event_type = "upload"  # or set dynamically based on your event trigger
    except KeyError as e:
        logger.error(f"Error parsing S3 event: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps(f"Error parsing S3 event: {str(e)}")
        }

    # Set up the payload for the FastAPI endpoint
    payload = {
        "bucket_name": bucket_name,
        "file_key": file_key,
        "event_type": event_type
    }

    # Define the FastAPI endpoint URL
    fastapi_url = "http://192.168.3.50:8510/process_s3_event"

    # Measure time for FastAPI request
    start_time = time.time()
    status = "Passed"  # Default status if request is sent successfully
    action = "upload" if event_type == "upload" else "delete"

    # Make the HTTP POST request to the FastAPI endpoint
    try:
        response = requests.post(fastapi_url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling FastAPI endpoint: {e}")
        status = "Failure"  # Update status if there was an error
    finally:
        elapsed_time = time.time() - start_time
        # Send Notion notification with status
        send_notion_notification(filename=file_key, action=action, status=status, elapsed_time=elapsed_time)
    
    # Return the Lambda function status
    return {
        "statusCode": 200,
        "body": json.dumps(f"Lambda function completed with status: {status}")
    }