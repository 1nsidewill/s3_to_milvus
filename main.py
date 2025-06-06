from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import os
import requests
import asyncio
from typing import List, Tuple
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
from tempfile import NamedTemporaryFile
from bs4 import BeautifulSoup

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

AWS_S3_BUCKET_NAME = "aiu-dev-data-bucket"
CHUNK_PARAM_S3_KEY = "Milvus/chunk_params.txt"

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

@app.post("/upload_parse_document")
async def upload_and_parse_document(file: UploadFile = File(...)):
    filename = file.filename.lower()
    ext = os.path.splitext(filename)[1]

    if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported.")

    ocr_mode = "auto" if ext == ".pdf" else "force"

    try:
        # 파일을 임시 저장
        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file_path = tmp.name
            content = await file.read()
            tmp.write(content)

        compressed_path = None

        if ext == ".pdf":
            is_text = is_text_based_pdf(file_path)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if not is_text and size_mb > 40:
                compressed_path = file_path.replace(".pdf", "_compressed.pdf")
                compressed_path = await compress_image_pdf(file_path, compressed_path)
            target_path = compressed_path if compressed_path else file_path
        else:
            target_path = file_path

        with open(target_path, "rb") as f:
            files = {"document": (os.path.basename(target_path), f, file.content_type)}
            data = {
                "output_formats": "['html']",
                "ocr": ocr_mode,
                "coordinates": "false",
                "model": "document-parse"
            }
            headers = {"Authorization": f"Bearer {upstage_api_key}"}
            response = requests.post(upstage_api_url, headers=headers, files=files, data=data)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Document parse failed")

        html = response.json().get("content", {}).get("html")
        parsed_html = clean_html_to_table_only(html)

        return parsed_html

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if compressed_path and os.path.exists(compressed_path):
                os.remove(compressed_path)
        except Exception as e:
            logger.warning(f"❌ 임시 파일 삭제 실패: {e}")


def clean_html_to_table_only(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 1. 테이블 이외 태그 제거
    for tag in soup.find_all():
        if tag.name not in ["table", "tr", "td", "tbody", "thead"]:
            tag.unwrap()

    # 2. 테이블 관련 태그 속성 제거
    for tag in soup.find_all(["table", "tr", "td", "tbody", "thead"]):
        tag.attrs = {}

    # 3. 스타일 태그 추가 (줄바꿈 제거, 문자열 직접 append)
    style_tag = soup.new_tag("style")
    style_tag.append("table, th, td { border: 1px solid black; border-collapse: collapse; padding: 4px; }")
    soup.insert(0, style_tag)

    return str(soup)


@app.post("/process_s3_event")
async def process_s3_event(event: S3Event, background_tasks: BackgroundTasks):
    """
    {
        "bucket_name": "aiu-dev-data-bucket",
        "file_key": "secondary_service/학생부종합전형 평가요소.txt",
        "event_type": "upload"
    }
    """
    logger.info(f"📩 Received S3 event: {event}")

    if event.file_key.endswith('/'):
        logger.info("📁 Folder creation event detected — skipping")
        return {"status": "Folder creation event ignored"}

    prefix_name = event.file_key.split('/')[1]
    filename = urllib.parse.unquote_plus(event.file_key.split('/')[-1])
    file_date = get_file_modified_date(event.bucket_name, event.file_key).isoformat()

    logger.info(f"📄 File info extracted — prefix: '{prefix_name}', filename: '{filename}', date: '{file_date}'")

    metadata = DocumentMetadata(
        filename=filename,
        file_date=str(file_date),
        collection_name=prefix_name
    )

    event_type = "upload" if event.event_type == "upload" else "delete"
    logger.info(f"📝 Event type resolved: {event_type}")

    if event_type == "upload":
        # ✅ 중복 방지 체크
        if has_collection(prefix_name):
            collection = Collection(prefix_name)
            collection.load()
            expr = f'file_name == "{filename}"'
            existing = collection.query(expr, output_fields=["file_name"])
            if existing:
                logger.info(f"⏩ Duplicate detected — skipping background task for '{filename}' in '{prefix_name}'")
                return {"status": "Duplicate — already processed"}

        else:
            logger.info(f"📦 Milvus collection '{prefix_name}' not found — creating new collection")
            create_milvus_collection(prefix_name)

        logger.info("🚀 Adding background task for document processing")
        background_tasks.add_task(handle_document_processing, event.bucket_name, event.file_key, metadata)
        return {"status": "Processing initiated"}

    elif event_type == "delete":
        logger.info("🗑️ Adding background task for deletion from Milvus")
        background_tasks.add_task(delete_from_milvus, metadata)
        return {"status": "Deletion initiated"}

    else:
        logger.warning(f"⚠️ Invalid event type received: {event.event_type}")
        raise HTTPException(status_code=400, detail="Invalid event type")

def get_file_modified_date(bucket_name, file_key):
    decoded_key = urllib.parse.unquote_plus(file_key)
    logger.debug(f"📅 Fetching modified date for: bucket='{bucket_name}', key='{decoded_key}'")

    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=decoded_key)
        logger.info("✅ File metadata fetched successfully")
        return response['LastModified']
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"❌ Object not found in bucket '{bucket_name}' with key '{decoded_key}'")
        else:
            logger.error(f"💥 Error retrieving object metadata: {e}")
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
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1}}
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
        # 📥 Load chunk size and overlap from config
        chunk_size, chunk_overlap = await get_chunk_params_from_s3(
            metadata.collection_name, default_chunk_size, default_chunk_overlap
        )
        logger.info(f"📁 Collection: {metadata.collection_name} | 🔧 Chunk Size: {chunk_size} | 🔁 Overlap: {chunk_overlap}")

        # ✂️ Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(document_html)
        logger.info(f"📄 Text split into {len(chunks)} chunks")

        # 🧠 Generate embeddings
        dense_embeddings, successful_chunks = generate_dense_embeddings(chunks)
        # sparse_embeddings = generate_sparse_embeddings(chunks)
        logger.info("🧬 Embeddings generated successfully")

        # 🔍 Ensure data consistency - use successful chunks
        if len(dense_embeddings) != len(successful_chunks):
            logger.error(f"❌ Embedding count ({len(dense_embeddings)}) doesn't match successful chunk count ({len(successful_chunks)})")
            return False

        if len(dense_embeddings) == 0:
            logger.error("❌ No embeddings were generated successfully")
            return False

        logger.info(f"📊 Using {len(successful_chunks)} successful chunks out of {len(chunks)} original chunks")

        # 📦 Prepare data for insertion
        insert_data = [
            dense_embeddings,
            # sparse_embeddings, 
            [metadata.filename] * len(successful_chunks), 
            [metadata.file_date] * len(successful_chunks), 
            successful_chunks
        ]

        # 🚀 Insert into Milvus
        collection = Collection(metadata.collection_name)
        logger.info(f"📤 Inserting {len(successful_chunks)} records into Milvus collection: {metadata.collection_name}")
        collection.insert(insert_data)
        collection.flush()

        # 🧱 Create index if needed and load
        create_index_if_not_exists(collection)
        collection.load()
        logger.info("✅ Data inserted and indexed in Milvus successfully")

        return True
    except Exception as e:
        logger.error(f"❌ Error inserting data into Milvus: {e}")
        return False

async def get_chunk_params_from_s3(collection_name: str, default_chunk_size: int, default_chunk_overlap: int):
    try:
        logger.info(f"🔍 Searching chunk params for collection: '{collection_name}' from S3")

        # S3에서 파일 내용 읽기
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=CHUNK_PARAM_S3_KEY)
        content = response['Body'].read().decode('utf-8')

        for line_number, line in enumerate(content.strip().splitlines(), start=1):
            logger.debug(f"📄 Line {line_number}: {repr(line)}")
            parts = line.strip().split('|')

            if len(parts) != 3:
                logger.warning(f"⚠️ Line {line_number} skipped: expected 3 parts but got {len(parts)} -> {parts}")
                continue

            name, size_str, overlap_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
            logger.debug(f"🔎 Parsed -> name='{name}', size='{size_str}', overlap='{overlap_str}'")

            if name == collection_name.strip():
                chunk_size = int(size_str)
                chunk_overlap = int(overlap_str)
                logger.info(f"✅ Match found: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
                return chunk_size, chunk_overlap

        logger.warning(f"⚠️ No match found for '{collection_name}', using default values: size={default_chunk_size}, overlap={default_chunk_overlap}")
        return default_chunk_size, default_chunk_overlap

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise FileNotFoundError(f"File not found in S3: {CHUNK_PARAM_S3_KEY}")
        raise HTTPException(status_code=500, detail=f"S3 access error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error processing chunk parameters from S3: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading chunk parameters: {str(e)}")

    return default_chunk_size, default_chunk_overlap

def generate_dense_embeddings(texts: List[str]) -> Tuple[List[List[float]], List[str]]:
    logger.info(f"Generating dense embeddings for {len(texts)} text chunks")
    
    # OpenAI API has a limit of 300,000 tokens per request
    # Use a smaller batch size to be safe (approximately 6000 tokens per chunk)
    batch_size = 20  # Process 20 chunks at a time to stay well under the limit
    all_embeddings = []
    successful_chunks = []
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch_texts)} chunks)")
        
        try:
            batch_embeddings = embeddings_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            successful_chunks.extend(batch_texts)
            logger.info(f"✅ Successfully processed batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"❌ Error processing batch {i//batch_size + 1}: {e}")
            # If batch fails, try processing one by one
            for j, text in enumerate(batch_texts):
                try:
                    single_embedding = embeddings_model.embed_documents([text])
                    all_embeddings.extend(single_embedding)
                    successful_chunks.append(text)
                    logger.info(f"✅ Successfully processed individual chunk {i + j + 1}")
                except Exception as single_e:
                    logger.error(f"❌ Failed to process chunk {i + j + 1}: {single_e}")
                    # Skip this chunk and continue
                    continue
    
    logger.info(f"Generated {len(all_embeddings)} embeddings successfully out of {len(texts)} chunks")
    if len(all_embeddings) < len(texts):
        logger.warning(f"⚠️ {len(texts) - len(all_embeddings)} chunks failed to generate embeddings")
    
    return all_embeddings, successful_chunks

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