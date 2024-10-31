# S3_TO_MILVUS 기술문서

## How To Run

### 프로젝트 클론 및 환경 설정

1. **GitHub에서 코드 클론**
    - 다음 명령어로 GitHub 리포지토리에서 프로젝트를 클론합니다:
    
    ```bash
    git clone <https://github.com/1nsidewill/s3_to_milvus.git>
    cd s3_to_milvus
    ```
    
2. **Python 버전 확인**
    - Python 3.11.10 버전이 필요합니다. 아래 명령어로 Python 버전을 확인하세요:
    
    ```bash
    python3 --version
    ```
    
3. **Poetry 설치 및 가상 환경 설정**
    - 프로젝트 의존성 관리를 위해 **Poetry**를 설치합니다. Poetry가 없는 경우 다음 명령어로 설치하세요:
    
    ```bash
    curl -sSL <https://install.python-poetry.org> | python3 -
    ```
    
    - 가상 환경을 프로젝트 폴더 내부에 생성하도록 설정:
    
    ```bash
    poetry config virtualenvs.in-project true
    ```
    
4. **Poetry로 의존성 설치**
    - 다음 명령어를 통해 프로젝트의 모든 의존성을 설치합니다:
    
    ```bash
    poetry install
    ```
    
5. **Poetry 쉘 활성화**
    - 가상 환경을 활성화하여 프로젝트 환경에 들어갑니다:
    
    ```bash
    poetry shell
    ```
    
6. **FastAPI 서버 실행**
    - FastAPI 서버를 디버그 모드로 실행하여 테스트합니다:
    
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8510 --reload
    ```

# 프로젝트 개요: AI 문서 처리 자동화 시스템

본 시스템은 AWS 기반의 서버리스 아키텍처로, S3에 업로드되는 파일을 Lambda를 통해 자동 처리하여 문서 내용을 Milvus에 임베딩하여 저장하고, Notion에 기록을 남기는 **자동화 솔루션**입니다

---

## 1. 아키텍처 개요

- **Storage (AWS S3)** : `aiu-dev-data-bucket`
    
    파일 업로드/삭제 트리거
    
- **Lambda (Python 3.11)** : `ai_s3_to_fastapi`
    
    이벤트 처리 및 FastAPI 호출
    
- **FastAPI 서비스** : `s3_to_milvus` (Python 3.11.10, Poetry, 배포 IP: `192.168.3.50:8510`)
    
    파일 다운로드 및 OCR, 임베딩 생성, Milvus Interaction
    
- **Notification** : Notion API
    
    작업 상태 기록 및 알림
    

---

![image](https://github.com/user-attachments/assets/31136aac-2f03-4543-bf27-7a8dfe4e43af)


## 2. 프로세스 흐름

1. **파일 업로드 및 Lambda 트리거**
    - `aiu-dev-data-bucket` S3 버킷에 파일(PDF, 이미지 등)이 업로드되면, Lambda 함수 (`ai_s3_to_fastapi`)가 자동 트리거됩니다.
    - Lambda 함수는 업로드 이벤트에서 `bucket_name`, `file_key`, `event_type`(업로드/삭제)을 추출하여 FastAPI 서버 (`s3_to_milvus`)의 `/process_s3_event` 엔드포인트에 POST 요청을 보냅니다.
2. **FastAPI (s3_to_milvus) - 파일 처리 및 임베딩 생성**
    - **업로드 이벤트 처리**
        - 이벤트 타입이 `upload`인 경우, FastAPI 서버는 해당 파일을 S3에서 로컬로 다운로드하고, **Upstage Document Parse API**를 호출해 문서를 HTML 형식으로 변환합니다.
        - 변환된 HTML 파일은 **LangChain의 RecursiveCharacterTextSplitter**로 텍스트를 여러 청크로 나누어 관리합니다.
        - 청크 단위 텍스트는 LangChain의 **OpenAIEmbeddings** (모델: `text-embedding-3-large`)를 사용해 벡터로 임베딩됩니다.
    - **Milvus Collection 생성 및 임베딩 저장**
        - S3의 각 폴더명은 **Milvus Collection**으로 취급됩니다.
        - Collection이 없을 경우, `create_milvus_collection` 함수로 아래와 같은 스키마로 Collection을 생성합니다.
            
            ```python
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="file_date", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            ```
            
        - Collection이 이미 존재하면 해당 폴더의 파일을 불러와 임베딩 데이터를 저장합니다.
        - 추가로, `create_index_if_not_exists` 함수는 벡터 필드에 대해 L2 거리 기반의 IVF_FLAT 인덱스를 생성하여 검색 효율성을 높입니다.
3. **삭제 이벤트 처리**
    - Lambda에서 전달된 `delete` 이벤트 시, FastAPI는 S3에서 삭제된 파일의 `file_key`를 기준으로 Milvus에서 해당 레코드를 제거합니다.
4. **Notion API로 작업 기록**
    - 모든 작업의 성공 및 실패 여부는 **Notion API**를 통해 노션 데이터베이스에 기록됩니다.
    - 각 기록은 작업 파일명, 상태(Passed/Failed), 처리 시간, 작업 메시지로 저장됩니다.

---

## 3. 기대 효과

- **자동화된 문서 처리**: S3에 파일을 업로드하는 것만으로 문서 내용이 자동으로 임베딩 및 데이터베이스에 저장됩니다.
- **작업 모니터링**: 모든 작업 상태를 Notion에 기록하여 작업 내역을 체계적으로 관리하고 추적할 수 있습니다.

---
