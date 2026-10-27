"""
Automated Dataset Cleaning API (FastAPI + MongoDB).

Collections (from the team schema): users, jobs, cleaning_reports.
Each uploaded file becomes one document in `jobs`.

Requirements:
    pip install fastapi uvicorn python-multipart python-dotenv pymongo pandas openpyxl xlrd

Run:
    uvicorn main:app --reload
"""

import os
import secrets
import zipfile
import logging
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4
import certifi
import pandas as pd
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import (
    FastAPI, UploadFile, File, Depends, HTTPException,
    Request, Query, Security,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.errors import PyMongoError
from starlette.concurrency import run_in_threadpool

# ---------------- Configuration ----------------

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL")
if not MONGODB_URL:
    # Without this check, the Mongo client would quietly use localhost.
    raise RuntimeError("MONGODB_URL is not set. Add it to your .env file.")

DATABASE_NAME = os.getenv("DATABASE_NAME", "data_cleaning_db")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "25")) * 1024 * 1024
API_KEY = os.getenv("API_KEY")
DEFAULT_USER_EMAIL = os.getenv("DEFAULT_USER_EMAIL", "test@example.com")

# One fixed absolute folder, so uploads always land in the same place.
UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR") or Path(__file__).parent / "uploads"
).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# First bytes of real Excel files, to catch renamed or fake files.
SIGNATURES = {
    ".xlsx": b"PK\x03\x04",
    ".xls": b"\xd0\xcf\x11\xe0",
}

JobStatus = Literal["uploaded", "processing", "cleaned", "failed"]


# ---------------- Response models ----------------

class JobOut(BaseModel):
    id: str
    user_id: str
    file_name: str
    file_type: str
    status: JobStatus
    created_at: datetime
    columns: list[str] = []
    file_size_bytes: Optional[int] = None


class JobList(BaseModel):
    total: int
    skip: int
    limit: int
    jobs: list[JobOut]


# ---------------- Helpers ----------------

# Fields that stay on the server and are never sent to clients.
HIDDEN = {"stored_filename": 0}


def serialize_job(doc: dict) -> dict:
    """Turn a Mongo document into JSON-friendly values (ObjectId -> str)."""
    return {
        "id": str(doc["_id"]),
        "user_id": str(doc["user_id"]),
        "file_name": doc["file_name"],
        "file_type": doc["file_type"],
        "status": doc["status"],
        "created_at": doc["created_at"],
        "columns": doc.get("columns", []),
        "file_size_bytes": doc.get("file_size_bytes"),
    }


def parse_id(value: str) -> ObjectId:
    """Convert a string id from the URL into an ObjectId, or return a 422."""
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=422, detail="Invalid job id")


def read_head(content: bytes, extension: str) -> pd.DataFrame:
    """Parse the first rows of the file. Runs in a worker thread."""
    buffer = BytesIO(content)
    if extension == ".csv":
        return pd.read_csv(buffer, nrows=5)
    return pd.read_excel(buffer, nrows=5)


# ---------------- MongoDB lifecycle ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncMongoClient(MONGODB_URL, tlsCAFile=certifi.where())

    try:
        await client.admin.command("ping")
        db = client[DATABASE_NAME]
        app.state.mongo_client = client
        app.state.database = db

        # Indexes: faster lookups, and no duplicate emails or reports.
        await db.users.create_index("email", unique=True)
        await db.jobs.create_index("user_id")
        await db.jobs.create_index("created_at")
        await db.cleaning_reports.create_index("job_id", unique=True)

        # Every job needs a user_id. Until real login exists, use one
        # default user, created automatically the first time.
        user = await db.users.find_one_and_update(
            {"email": DEFAULT_USER_EMAIL},
            {"$setOnInsert": {
                "name": "Default User",
                "created_at": datetime.now(timezone.utc),
            }},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        app.state.default_user_id = user["_id"]

        logger.info("Connected to MongoDB")
        yield

    finally:
        await client.close()
        logger.info("MongoDB connection closed")


app = FastAPI(
    title="Automated Dataset Cleaning Pipeline",
    description="API for uploading and managing datasets",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------- Middleware ----------------

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Refuse oversized uploads early, using the Content-Length header.

    Also set a body limit at your proxy (e.g. nginx client_max_body_size);
    the header can be missing, so the check in the route stays as backup.
    """
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_FILE_SIZE + 1_000_000:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request too large"},
        )
    return await call_next(request)


# Added after the size limiter so it wraps it: 413 replies get CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONTEND_ORIGIN.split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ---------------- Dependencies ----------------

async def get_database(request: Request):
    """Inject the shared MongoDB database into routes."""
    return request.app.state.database


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_key(key: Optional[str] = Security(api_key_header)):
    """Only enforced when API_KEY is set in the environment."""
    if not API_KEY:
        return
    if not key or not secrets.compare_digest(key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


protected = [Depends(require_key)]


# ---------------- Error handling ----------------

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unexpected API error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------- Health check ----------------

@app.get("/")
async def home():
    return {"message": "Dataset Cleaning API is running"}


@app.get("/health")
async def health(db=Depends(get_database)):
    try:
        await db.command("ping")
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "healthy", "database": "connected"}


# ---------------- Upload (creates a job) ----------------

@app.post(
    "/jobs/upload",
    status_code=201,
    response_model=JobOut,
    dependencies=protected,
)
async def upload_job(
    request: Request,
    file: UploadFile = File(...),
    db=Depends(get_database),
):
    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(status_code=400, detail="A filename is required")

    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Only CSV, XLSX and XLS files are supported",
        )

    # Read at most MAX_FILE_SIZE + 1 bytes to detect oversized files.
    content = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the configured size limit",
        )

    # Excel files must start with the right signature bytes.
    signature = SIGNATURES.get(extension)
    if signature and not content.startswith(signature):
        raise HTTPException(
            status_code=422,
            detail="File content does not match its extension",
        )

    stored_name = f"{uuid4()}{extension}"
    file_path = UPLOAD_DIR / stored_name

    try:
        # Parse in a worker thread so other requests are not blocked.
        df = await run_in_threadpool(read_head, content, extension)

        if len(df.columns) == 0:
            raise HTTPException(status_code=422, detail="Dataset has no columns")

        await run_in_threadpool(file_path.write_bytes, content)

        document = {
            # Fields from the team schema
            "user_id": request.app.state.default_user_id,
            "file_name": original_name,
            "file_type": extension,
            "status": "uploaded",
            "created_at": datetime.now(timezone.utc),
            # Extra fields (confirm with your teammate that these are allowed)
            "stored_filename": stored_name,
            "columns": [str(c) for c in df.columns],
            "file_size_bytes": len(content),
        }

        result = await db.jobs.insert_one(document)
        document["_id"] = result.inserted_id
        return serialize_job(document)

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except (pd.errors.ParserError, ValueError, zipfile.BadZipFile):
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="Could not parse the dataset. Check its format.",
        )

    except PyMongoError:
        file_path.unlink(missing_ok=True)
        logger.exception("MongoDB operation failed")
        raise HTTPException(
            status_code=503,
            detail="Could not save job metadata",
        )

    except Exception:
        file_path.unlink(missing_ok=True)
        raise


# ---------------- List jobs ----------------

@app.get("/jobs", response_model=JobList, dependencies=protected)
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db=Depends(get_database),
):
    try:
        cursor = (
            db.jobs.find({}, HIDDEN)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        total = await db.jobs.count_documents({})

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "jobs": [serialize_job(d) for d in docs],
        }

    except PyMongoError:
        logger.exception("Could not list jobs")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")


# ---------------- Get one job ----------------

@app.get("/jobs/{job_id}", response_model=JobOut, dependencies=protected)
async def get_job(job_id: str, db=Depends(get_database)):
    oid = parse_id(job_id)

    try:
        doc = await db.jobs.find_one({"_id": oid}, HIDDEN)
    except PyMongoError:
        logger.exception("Could not retrieve job")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")

    if doc is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return serialize_job(doc)


# ---------------- Cleaning report for a job ----------------

@app.get("/jobs/{job_id}/report", dependencies=protected)
async def get_report(job_id: str, db=Depends(get_database)):
    """Returns the cleaning report once the pandas step has saved one."""
    oid = parse_id(job_id)

    try:
        report = await db.cleaning_reports.find_one({"job_id": oid})
    except PyMongoError:
        logger.exception("Could not retrieve report")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")

    if report is None:
        raise HTTPException(status_code=404, detail="No report for this job yet")

    report["id"] = str(report.pop("_id"))
    report["job_id"] = str(report["job_id"])
    return report


# ---------------- Delete a job ----------------

@app.delete("/jobs/{job_id}", dependencies=protected)
async def delete_job(job_id: str, db=Depends(get_database)):
    oid = parse_id(job_id)

    try:
        # One step: find and delete together, so two requests can't clash.
        doc = await db.jobs.find_one_and_delete({"_id": oid})

        if doc is None:
            raise HTTPException(status_code=404, detail="Job not found")

        await db.cleaning_reports.delete_many({"job_id": oid})

    except PyMongoError:
        logger.exception("Could not delete job")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")

    # The record is gone; a leftover file is only logged, not an error.
    try:
        (UPLOAD_DIR / doc["stored_filename"]).unlink(missing_ok=True)
    except OSError:
        logger.exception("Could not remove file %s", doc.get("stored_filename"))

    return {"message": "Job deleted successfully", "id": job_id}