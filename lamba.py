import json
import requests
import logging
import time
from datetime import datetime
from urllib.parse import unquote

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Notion API credentials
notion_token = ""  # Replace with your Notion integration token
notion_database_id = ""

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
    try:
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        event_name = event['Records'][0]['eventName']

        # 이벤트 타입 판단
        if "ObjectCreated" in event_name:
            event_type = "upload"
        elif "ObjectRemoved" in event_name:
            event_type = "delete"
        else:
            print(f"Unrecognized event type: {event_name}")
            return {
                "statusCode": 400,
                "body": json.dumps(f"Unrecognized event type: {event_name}")
            }

        # 디렉토리 생성 무시
        if file_key.endswith('/'):
            print("Folder creation detected; no action taken.")
            return {
                "statusCode": 200,
                "body": json.dumps("Folder creation detected; no action taken.")
            }

        decoded_file_key = unquote(file_key)

        # ✅ Milvus 디렉토리가 아닌 경우 무시
        if not decoded_file_key.startswith("Milvus/"):
            print(f"Ignored: Not under 'Milvus/' directory -> {decoded_file_key}")
            return {
                "statusCode": 200,
                "body": json.dumps("Not under 'Milvus/' directory; skipping.")
            }

        # ✅ Milvus 하위지만, 바로 아래 파일이면 (ex: Milvus/chunk_params.txt) 무시
        milvus_subpath = decoded_file_key[len("Milvus/"):]
        if '/' not in milvus_subpath:
            print(f"Ignored: File is directly under 'Milvus/' -> {decoded_file_key}")
            return {
                "statusCode": 200,
                "body": json.dumps("File directly under 'Milvus/'; skipping.")
            }

        # ✅ Milvus/collection_name/ 하위 파일만 통과
        print(f"✅ Processing event: bucket={bucket_name}, file_key={decoded_file_key}, event_type={event_type}")

    except KeyError as e:
        print(f"Error parsing S3 event: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps(f"Error parsing S3 event: {str(e)}")
        }

    # FastAPI 호출 payload
    payload = {
        "bucket_name": bucket_name,
        "file_key": file_key,
        "event_type": event_type
    }

    fastapi_url = "http://192.168.3.50/s3_to_milvus/process_s3_event"

    start_time = time.time()
    status = "Passed"
    action = event_type

    try:
        print('Trying to call FastAPI URL...')
        response = requests.post(fastapi_url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error calling FastAPI endpoint: {e}")
        status = "Failure"
    finally:
        elapsed_time = time.time() - start_time
        send_notion_notification(filename=decoded_file_key, action=action, status=status, elapsed_time=elapsed_time)

    return {
        "statusCode": 200,
        "body": json.dumps(f"Lambda function completed with status: {status}")
    }
