from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os
import requests
import asyncio
from typing import List
import boto3
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, has_collection
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from datetime import datetime
import logging
import time
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import pdfplumber
from pdf2image import convert_from_path
import aiofiles
import urllib.parse
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
milvus_url = os.getenv("MILVUS_URL")
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
notion_token = os.getenv("NOTION_TOKEN")
notion_database_id = os.getenv("NOTION_DATABASE_ID")
upstage_api_key = os.getenv("UPSTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
upstage_api_url = os.getenv("UPSTAGE_API_URL")

# Configure AWS S3 and Milvus connections
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
connections.connect(host=milvus_url.split(":")[0], port=milvus_url.split(":")[1])

app = FastAPI()

@app.get("/", include_in_schema=False)
async def root(request: Request):
    return RedirectResponse(url=request.scope.get("root_path") + "/docs")

class S3Event(BaseModel):
    bucket_name: str
    file_key: str
    event_type: str  # "upload" or "delete"

class DocumentMetadata(BaseModel):
    filename: str
    file_date: str
    collection_name: str

@app.post("/process_s3_event")
async def process_s3_event(event: S3Event, background_tasks: BackgroundTasks):
    """
    {
        "bucket_name": "aiu-dev-data-bucket",
        "file_key": "secondary_service/학생부종합전형 평가요소.txt",
        "event_type": "upload"
    }
    """
    if event.file_key.endswith('/'):
        return {"status": "Folder creation event ignored"}

    prefix_name = event.file_key.split('/')[1]
    filename = urllib.parse.unquote_plus(event.file_key.split('/')[-1])
    file_date = get_file_modified_date(event.bucket_name, event.file_key).isoformat()
    metadata = DocumentMetadata(filename=filename, file_date=str(file_date), collection_name=prefix_name)
    event_type = "upload" if event.event_type == "upload" else "delete"

    if event_type == "upload":
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
    decoded_key = urllib.parse.unquote_plus(file_key)
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=decoded_key)
        return response['LastModified']
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"Object not found in bucket '{bucket_name}' with key '{decoded_key}'")
        else:
            logger.error(f"An error occurred: {e}")
        return datetime.now()

def create_milvus_collection(collection_name: str):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072),
        # FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="file_date", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields=fields, description=collection_name, enable_dynamic_field=True)
    Collection(name=collection_name, schema=schema)
    logger.info(f"Milvus collection '{collection_name}' created.")

def create_index_if_not_exists(collection: Collection):
    if not collection.indexes:
        logger.info("Creating index...")
        collection.create_index(
            field_name="vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 20}}
        )
        logger.info("Dense Index created successfully.")

        # collection.create_index(
        #     field_name="sparse_vector",
        #     index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        # )
        # logger.info("Sparse created successfully.")

async def handle_document_processing(bucket_name: str, file_key: str, metadata: DocumentMetadata):
    start_time = time.time()
    logger.info(f"Started processing for file '{file_key}' in bucket '{bucket_name}'")

    file_path = await download_file_from_s3(bucket_name, file_key)
    if not file_path:
        logger.error("File download failed. Skipping further processing for %s/%s", bucket_name, file_key)
        await send_notion_notification(metadata, "processed", 0, False)
        return {"status": "File not found, processing skipped"}

    logger.info(f"File successfully downloaded to '{file_path}'")
    
    # Initialize compressed_file_path
    compressed_file_path = None

    # Determine file type and process accordingly
    file_extension = os.path.splitext(file_path)[1].lower()
    document_html = None
    success = False

    if file_extension == '.pdf':
        is_text_based = is_text_based_pdf(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        MAX_SIZE_MB = 40

        if not is_text_based and file_size_mb > MAX_SIZE_MB:
            compressed_file_path = file_path.replace(".pdf", "_compressed.pdf")
            compressed_file_path = await compress_image_pdf(file_path, compressed_file_path)

        target_file_path = compressed_file_path if compressed_file_path else file_path
        ocr_mode = "auto" if is_text_based else "force"

        with open(target_file_path, "rb") as temp_file:
            files = {"document": (os.path.basename(target_file_path), temp_file, "application/pdf")}
            body = {
                "output_formats": "['html']",
                "ocr": ocr_mode,
                "coordinates": "false",
                "model": "document-parse"
            }

            headers = {"Authorization": f"Bearer {upstage_api_key}"}
            response = requests.post(upstage_api_url, headers=headers, files=files, data=body)

        if response.status_code == 200:
            response_data = response.json()
            document_html = response_data.get('content', {}).get('html', None)
            if document_html:
                success = await insert_to_milvus(document_html, metadata)

    elif file_extension in ['.png', '.jpg', '.jpeg']:
        # Force OCR for image files
        ocr_mode = "force"
        with open(file_path, "rb") as temp_file:
            files = {"document": (os.path.basename(file_path), temp_file, "application/octet-stream")}
            body = {
                "output_formats": "['html']",
                "ocr": ocr_mode,
                "coordinates": "false",
                "model": "document-parse"
            }

            headers = {"Authorization": f"Bearer {upstage_api_key}"}
            response = requests.post(upstage_api_url, headers=headers, files=files, data=body)

        if response.status_code == 200:
            response_data = response.json()
            document_html = response_data.get('content', {}).get('html', None)
            if document_html:
                success = await insert_to_milvus(document_html, metadata)

    elif file_extension == '.txt':
        # Directly insert text file content into Milvus
        with open(file_path, 'r', encoding='utf-8') as text_file:
            text_content = text_file.read()
        success = await insert_to_milvus(text_content, metadata)

    else:
        logger.error("Unsupported file type for %s", file_path)
        await send_notion_notification(metadata, "processed", 0, False)
        return {"status": "Unsupported file type, processing skipped"}

    if not success:
        logger.error("Processing failed for file '%s'", file_key)
        raise HTTPException(status_code=400, detail="Processing failed")

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"Completed processing for '{file_key}'. Total time: {elapsed_time} seconds")
    await send_notion_notification(metadata, "processed", elapsed_time, success)
    
    # Cleanup: Delete downloaded files
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temporary file '{file_path}'")
        if compressed_file_path and os.path.exists(compressed_file_path):
            os.remove(compressed_file_path)
            logger.info(f"Deleted temporary file '{compressed_file_path}'")
    except Exception as e:
        logger.error(f"Error deleting temporary files: {e}")

    return {"status": "Processing completed", "elapsed_time": elapsed_time, "success": success}

async def download_file_from_s3(bucket_name: str, file_key: str) -> str:
    # Decode the file key (handles '+' as space)
    decoded_key = urllib.parse.unquote_plus(file_key)
    local_file_path = f"data/{decoded_key}"
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    logger.info(f"Attempting to download file '{decoded_key}' from bucket '{bucket_name}' to '{local_file_path}'")
    
    try:
        # Download the file from S3
        s3_client.download_file(bucket_name, decoded_key, local_file_path)
        logger.info(f"File downloaded successfully: {local_file_path}")
        return local_file_path
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"Object not found in bucket '{bucket_name}' with key '{decoded_key}'")
        else:
            logger.error(f"An error occurred: {e}")
        return None

def is_text_based_pdf(file_path) -> bool:
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                return True
    return False

async def compress_image_pdf(input_pdf: str, output_pdf: str) -> str:
    try:
        images = await asyncio.to_thread(convert_from_path, input_pdf)
        compressed_images = []

        for img in images:
            compressed_image = await asyncio.to_thread(img.convert, "RGB")
            compressed_images.append(compressed_image)

        await asyncio.to_thread(
            compressed_images[0].save,
            output_pdf,
            save_all=True,
            append_images=compressed_images[1:],
            quality=70,
        )

        return output_pdf
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 압축 중 오류 발생: {str(e)}")
    
async def insert_to_milvus(document_html: str, metadata: DocumentMetadata, default_chunk_size: int = 3000, default_chunk_overlap: int = 300) -> bool:
    try:
        # Load chunk size and overlap values from TXT
        chunk_size, chunk_overlap = get_chunk_params_from_txt(metadata.collection_name, default_chunk_size, default_chunk_overlap)

        logger.info(f"Collection Name: {metadata.collection_name}, Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(document_html)
        logger.info(f"Split OCR text into {len(chunks)} chunks")

        dense_embeddings = generate_dense_embeddings(chunks)
        # sparse_embeddings = generate_sparse_embeddings(chunks)
        logger.info("Embeddings generated, preparing data for Milvus insertion")

        insert_data = [
            dense_embeddings,
            # sparse_embeddings, 
            [metadata.filename] * len(chunks), 
            [metadata.file_date] * len(chunks), 
            chunks
        ]

        collection = Collection(metadata.collection_name)
        logger.info(f"Inserting data into Milvus collection: {metadata.collection_name}")
        collection.insert(insert_data)
        collection.flush()
        create_index_if_not_exists(collection)
        collection.load()
        logger.info("Data inserted and indexed in Milvus successfully")
        return True
    except Exception as e:
        logger.error(f"Error inserting data into Milvus: {e}")
        return False

def get_chunk_params_from_txt(collection_name: str, default_chunk_size: int, default_chunk_overlap: int):
    try:
        with open('chunk_params.txt', 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) == 3 and parts[0] == collection_name:
                    return int(parts[1]), int(parts[2])
    except Exception as e:
        logger.error(f"Error reading chunk parameters from TXT: {e}")
    
    # Return default values if no match is found or an error occurs
    return default_chunk_size, default_chunk_overlap

def generate_dense_embeddings(texts: List[str]) -> List[List[float]]:
    logger.info("Generating dense embeddings for text chunks")
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large").embed_documents(texts)

def generate_sparse_embeddings(texts: List[str]) -> List[List[float]]:
    logger.info("Generating sparse embeddings for text chunks")
    return BM25SparseEmbedding(corpus=texts).embed_documents()

async def delete_from_milvus(metadata: DocumentMetadata):
    start_time = time.time()
    
    if not has_collection(metadata.collection_name):
        logger.warning(f"Collection '{metadata.collection_name}' does not exist. Skipping deletion.")
        elapsed_time = round(time.time() - start_time, 2)
        await send_notion_notification(metadata, "deletion attempted", elapsed_time, False)
        return {"status": "Collection not found; deletion skipped"}

    collection = Collection(metadata.collection_name)
    expr = f'file_name == "{metadata.filename}"'
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
        "Notion-Version": "2022-06-28"
    }
    
    title = metadata.collection_name + "/" + metadata.filename
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