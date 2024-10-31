from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os
import requests
import asyncio
import aiofiles
from typing import List
import boto3
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, has_collection
from PyPDF2 import PdfReader, PdfWriter
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from datetime import datetime
import logging
import time
from bs4 import BeautifulSoup
from urllib.parse import unquote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure AWS S3 and Milvus connections
milvus_url = os.getenv("MILVUS_URL")
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
notion_token = os.getenv("NOTION_TOKEN")
notion_database_id = os.getenv("NOTION_DATABASE_ID")
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
connections.connect(host=milvus_url.split(":")[0], port=milvus_url.split(":")[1])

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

"""
{
    "bucket_name": "aiu-dev-data-bucket",
    "file_key": "korea_university/02-문과대학-01.국어국문.pdf",
    "event_type": "upload"
}
"""
class S3Event(BaseModel):
    bucket_name: str
    file_key: str
    event_type: str  # "upload" or "delete"

class DocumentMetadata(BaseModel):
    filename: str
    file_date: str
    collection_name: str

# Endpoint to process S3 events
@app.post("/process_s3_event")
async def process_s3_event(event: S3Event, background_tasks: BackgroundTasks):
    # Check if the event is for a folder creation and skip processing
    if event.file_key.endswith('/'):
        return {"status": "Folder creation event ignored"}

    # Split and decode the file key
    prefix_name = event.file_key.split('/')[0]
    filename = unquote(event.file_key.split('/')[-1])
    
    # Get modified date and convert to string for metadata
    file_date = get_file_modified_date(event.bucket_name, event.file_key).isoformat()

    # Prepare metadata as a Pydantic model
    metadata = DocumentMetadata(filename=filename, file_date=str(file_date), collection_name=prefix_name)

    # Determine the event type dynamically
    event_type = "upload" if event.event_type == "ObjectCreated" else "delete"  # Adjust as needed

    # Initiate background tasks based on event type
    if event_type == "upload":
        # Check if collection exists and create if it does not
        if not has_collection(prefix_name):
            create_milvus_collection(prefix_name)

        background_tasks.add_task(handle_document_processing, event.bucket_name, event.file_key, metadata)
        return {"status": "Processing initiated"}
    elif event_type == "delete":
        background_tasks.add_task(delete_from_milvus, metadata)
        return {"status": "Deletion initiated"}
    else:
        raise HTTPException(status_code=400, detail="Invalid event type")

def get_file_modified_date(bucket_name, file_key):
    decoded_key = unquote(file_key)  # Decode URL-encoded characters
    response = s3_client.head_object(Bucket=bucket_name, Key=decoded_key)
    return response['LastModified']

def create_milvus_collection(collection_name: str):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="file_date", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields=fields, description=collection_name, enable_dynamic_field=True)
    Collection(name=collection_name, schema=schema)
    print(f"Milvus collection '{collection_name}' created.")

def create_index_if_not_exists(collection: Collection):
    if not collection.indexes:
        print("Creating index...")
        collection.create_index(
            field_name="vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 16}}
        )
        print("Index created successfully.")

async def handle_document_processing(bucket_name: str, file_key: str, metadata: DocumentMetadata):
    start_time = time.time()
    file_path = await download_file_from_s3(bucket_name, file_key)
    parsed_data_directory = os.path.join(os.path.dirname(file_path), "parsed_data")
    os.makedirs(parsed_data_directory, exist_ok=True)

    if os.path.getsize(file_path) > 50 * 1024 * 1024:
        split_files = split_pdf(file_path)
        for split_file in split_files:
            output_path = os.path.join(parsed_data_directory, f"{os.path.basename(split_file)}_parsed")
            success = await call_upstage_api(split_file, output_path)
            if success:
                await insert_to_milvus(output_path, metadata)
            else:
                print(f"Skipping Milvus insertion for {split_file} due to OCR failure.")
    else:
        output_path = os.path.join(parsed_data_directory, f"{os.path.basename(file_path)}")
        success = await call_upstage_api(file_path, output_path)
        if success:
            await insert_to_milvus(output_path, metadata)
        else:
            print(f"Skipping Milvus insertion for {file_path} due to OCR failure.")
    
    elapsed_time = round(time.time() - start_time, 2)
    await send_notion_notification(metadata, "processed", elapsed_time, success)
    
async def download_file_from_s3(bucket_name: str, file_key: str) -> str:
    local_file_path = f"data/{file_key}"
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    s3_client.download_file(bucket_name, file_key, local_file_path)
    logger.info("File downloaded successfully: %s", local_file_path)
    return local_file_path

def split_pdf(input_file: str) -> List[str]:
    split_files = []
    reader = PdfReader(input_file)
    total_pages = len(reader.pages)

    output_dir = "split_data"
    os.makedirs(output_dir, exist_ok=True)

    pages_per_chunk = 10
    for start_page in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        end_page = min(start_page + pages_per_chunk, total_pages)
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        split_file_path = os.path.join(output_dir, f"{os.path.basename(input_file)}_part_{start_page//pages_per_chunk + 1}.pdf")
        with open(split_file_path, "wb") as output_pdf:
            writer.write(output_pdf)
        split_files.append(split_file_path)

    return split_files

async def call_upstage_api(input_file: str, output_file: str) -> bool:
    UPSTAGE_API_URL = os.getenv("UPSTAGE_API_URL")
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}

    with open(input_file, "rb") as file_data:
        response = requests.post(UPSTAGE_API_URL, headers=headers, files={"document": file_data}, data={"ocr": "force"})
    if response.status_code == 202:
        return await poll_for_result(response.json()["request_id"], output_file)
    else:
        logger.error(f"Error processing file {input_file}: {response.text}")
        return False

async def poll_for_result(request_id: str, output_file: str) -> bool:
    while True:
        response = requests.get(f"https://api.upstage.ai/v1/document-ai/requests/{request_id}", headers={"Authorization": f"Bearer {os.getenv('UPSTAGE_API_KEY')}"})
        if response.status_code == 200 and response.json()["status"] == "completed":
            await download_inference_result(response.json()["batches"][0]["download_url"], output_file)
            return True
        elif response.status_code != 200 or response.json()["status"] == "failed":
            return False
        await asyncio.sleep(3)

async def download_inference_result(download_url: str, output_file: str):
    response = requests.get(download_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.json()['content']['html'], "html.parser")
        for img_tag in soup.find_all('img'):
            if img_tag.has_attr('alt'):
                img_tag.insert_before(img_tag['alt'])
            img_tag.decompose()
        clean_text = soup.get_text(strip=True)
        async with aiofiles.open(output_file + ".html", 'w') as output:
            await output.write(clean_text)

async def insert_to_milvus(output_file: str, metadata: DocumentMetadata, chunk_size: int = 1000, chunk_overlap: int = 100):
    # Read OCR output file
    async with aiofiles.open(output_file + ".html", 'r') as file:
        ocr_text = await file.read()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(ocr_text)
    
    # Generate embeddings for all chunks
    embeddings = generate_embeddings(chunks)
    insert_data = [
        embeddings, 
        [metadata.filename] * len(chunks), 
        [metadata.file_date] * len(chunks), 
        chunks
    ]

    # Insert data into Milvus
    collection = Collection(metadata.collection_name)
    collection.insert(insert_data)
    collection.flush()
    create_index_if_not_exists(collection)
    collection.load()

    # Delete the original file after successful insertion
    try:
        os.remove(output_file + ".html")
        print(f"File '{output_file}.html' deleted successfully after insertion.")
    except Exception as e:
        print(f"Error deleting file '{output_file}.html': {e}")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-large").embed_documents(texts)

async def delete_from_milvus(metadata: DocumentMetadata):
    start_time = time.time()
    
    # Check if the collection exists before attempting deletion
    if not has_collection(metadata.collection_name):
        logger.warning(f"Collection '{metadata.collection_name}' does not exist. Skipping deletion.")
        elapsed_time = round(time.time() - start_time, 2)
        await send_notion_notification(metadata, "deletion attempted", elapsed_time, False)
        return {"status": "Collection not found; deletion skipped"}

    # Proceed with deletion if collection exists
    collection = Collection(metadata.collection_name)
    expr = f'file_name == "{metadata.filename}" && file_date == "{metadata.file_date}"'
    collection.delete(expr)
    collection.load()

    elapsed_time = round(time.time() - start_time, 2)
    await send_notion_notification(metadata, "deleted", elapsed_time, True)
    return {"status": "Deletion completed"}


async def send_notion_notification(metadata: DocumentMetadata, action: str, elapsed_time: float, success: bool):
    status = "Success" if success else "Failure"
    message = f"The file '{metadata.filename}' was {action} with {status} in {elapsed_time:.2f} seconds."
    notification_time = datetime.now().isoformat()
    
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"  # Notion API version
    }
    
    title = metadata.collection_name + "/" + metadata.filename
    # Data payload for Notion APIa
    data = {
        "parent": {"database_id": notion_database_id},
        "properties": {
            "Title": {
                "title": [{"text": {"content": title}}]
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