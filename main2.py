import os
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from io import BytesIO, StringIO
import asyncio
from dotenv import load_dotenv
# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Data processing imports
import pandas as pd
import anthropic


# Import your processors
from payroll_extractor import (
    find_employees_enhanced,
    extract_enhanced_payroll_records,
    process_enhanced_payroll,
    save_results as save_payroll_results
)
from timesheet_extractor import (
    find_employees,
    extract_employee_records_with_continuation,
    process_timesheet,
    save_results as save_timesheet_results
)
from all_imports import (
    EnhancedOCRProcessor,
    ProcessingConfig,
    ProcessingResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# ===========================
# CONFIGURATION & SETUP
# ===========================

class APIConfig:
    """API Configuration"""
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
    MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '5'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '300'))

    # API Keys from environment
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable required")


# Initialize FastAPI app
app = FastAPI(
    title="Production Document Processing & Employee Data Extraction API",
    description="Complete API for document processing with timesheet and payroll data extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files directory (for HTML, CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global processors
ocr_processor = None
processing_config = None

# Job tracking
active_jobs: Dict[str, Dict] = {}


# ===========================
# PYDANTIC MODELS
# ===========================

class ProcessingResponse(BaseModel):
    """Base processing response model"""
    success: bool
    message: str
    processing_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class DocumentProcessingResponse(ProcessingResponse):
    """Document processing specific response"""
    filename: str
    file_type: str
    file_size: int
    content_length: Optional[int] = None
    pages_processed: Optional[int] = None


class TimesheetExtractionResponse(ProcessingResponse):
    """Basic timesheet extraction response"""
    filename: str
    file_type: str
    employee_names: List[str] = []
    total_employees: int = 0
    total_records: int = 0
    employee_records: List[Dict[str, Any]] = []
    extraction_summary: Optional[Dict[str, Any]] = None


class PayrollExtractionResponse(ProcessingResponse):
    """Enhanced payroll extraction response"""
    filename: str
    file_type: str
    employee_names: List[str] = []
    total_employees: int = 0
    total_records: int = 0
    employee_records: List[Dict[str, Any]] = []
    payroll_summary: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    active_jobs: int
    system_info: Dict[str, Any]


# ===========================
# STARTUP & SHUTDOWN
# ===========================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ocr_processor, processing_config

    try:
        # Validate configuration
        APIConfig.validate_config()

        # Initialize OCR processor
        processing_config = ProcessingConfig.from_env()
        ocr_processor = EnhancedOCRProcessor(
            api_key=APIConfig.OPENAI_API_KEY,
            config=processing_config
        )

        logger.info("✅ Complete API Server started successfully")
        logger.info(f"📊 Max file size: {APIConfig.MAX_FILE_SIZE_MB}MB")
        logger.info(f"🔄 Max concurrent jobs: {APIConfig.MAX_CONCURRENT_JOBS}")
        logger.info("🎯 Available endpoints: /extract-timesheet, /extract-payroll, /process-document")
        logger.info("🌐 Frontend UI available at: / (root)")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 API Server shutting down...")
    global active_jobs
    active_jobs.clear()
    logger.info("✅ API Server shutdown complete")


# ===========================
# UTILITY FUNCTIONS
# ===========================

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Check file extension
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.csv', '.xlsx', '.xls'}
    file_ext = '.' + file.filename.lower().split('.')[-1]

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )


def get_file_type(filename: str) -> str:
    """Get standardized file type"""
    ext = filename.lower().split('.')[-1]

    type_mapping = {
        'pdf': 'pdf',
        'png': 'image',
        'jpg': 'image',
        'jpeg': 'image',
        'csv': 'csv',
        'xlsx': 'excel',
        'xls': 'excel'
    }

    return type_mapping.get(ext, 'unknown')


async def process_document_content(file_content: bytes, filename: str) -> str:
    """Process document content based on file type"""
    file_type = get_file_type(filename)

    try:
        if file_type == 'csv':
            # Process CSV
            try:
                content_str = file_content.decode("utf-8")
            except UnicodeDecodeError:
                content_str = file_content.decode("latin-1")

            df = pd.read_csv(StringIO(content_str))
            return df.to_string()


        elif file_type == 'excel':
            file_obj = BytesIO(file_content)
            try:
                df = pd.read_excel(file_obj, engine='openpyxl', sheet_name=0)
            except:
                try:
                    file_obj.seek(0)
                    df = pd.read_excel(file_obj, engine='xlrd', sheet_name=0)
                except:
                    file_obj.seek(0)
                    df = pd.read_excel(file_obj, sheet_name=0)
            return df.to_string()

        elif file_type == 'pdf':
            # Process PDF with OCR
            result = ocr_processor.process_pdf_document(file_content, filename)
            if not result.success:
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {result.error}")
            return result.content

        elif file_type == 'image':
            # Process Image with OCR
            result = ocr_processor.process_image_document(file_content, filename)
            if not result.success:
                raise HTTPException(status_code=500, detail=f"Image processing failed: {result.error}")
            return result.content

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


def generate_processing_id() -> str:
    """Generate unique processing ID"""
    return f"proc_{int(time.time())}_{os.urandom(4).hex()}"


def track_job(processing_id: str, job_info: Dict) -> None:
    """Track active job"""
    active_jobs[processing_id] = {
        **job_info,
        'start_time': time.time(),
        'status': 'processing'
    }


def complete_job(processing_id: str, success: bool = True) -> None:
    """Mark job as complete"""
    if processing_id in active_jobs:
        active_jobs[processing_id]['status'] = 'completed' if success else 'failed'
        active_jobs[processing_id]['end_time'] = time.time()

        # Clean up old jobs (keep last 100)
        if len(active_jobs) > 100:
            oldest_jobs = sorted(active_jobs.items(), key=lambda x: x[1]['start_time'])[:50]
            for job_id, _ in oldest_jobs:
                del active_jobs[job_id]


# ===========================
# FRONTEND ENDPOINTS
# ===========================

@app.get("/")
async def serve_frontend():
    """Serve the main HTML frontend"""
    return FileResponse('static/index.html')


@app.get("/app")
async def serve_app():
    """Alternative endpoint for the frontend"""
    return FileResponse('static/index.html')


# ===========================
# API ENDPOINTS
# ===========================

@app.get("/api", response_model=Dict[str, Any])
async def api_root():
    """API root endpoint with information"""
    return {
        "service": "Complete Document Processing & Employee Data Extraction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "process_document": "/process-document",
            "extract_timesheet": "/extract-timesheet (basic employee timesheet data)",
            "extract_payroll": "/extract-payroll (comprehensive payroll data with 28+ fields)",
            "job_status": "/job-status/{processing_id}"
        },
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "CSV", "XLSX", "XLS"],
        "extraction_types": {
            "timesheet": "Basic employee timesheet extraction (14 fields)",
            "payroll": "Enhanced payroll extraction (28+ fields with taxes & deductions)"
        },
        "max_file_size": f"{APIConfig.MAX_FILE_SIZE_MB}MB",
        "documentation": "/docs",
        "frontend": "/"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        active_jobs=len(active_jobs),
        system_info={
            "max_file_size_mb": APIConfig.MAX_FILE_SIZE_MB,
            "max_concurrent_jobs": APIConfig.MAX_CONCURRENT_JOBS,
            "ocr_processor_ready": ocr_processor is not None,
            "api_keys_configured": bool(APIConfig.ANTHROPIC_API_KEY and APIConfig.OPENAI_API_KEY),
            "supported_extractions": ["timesheet", "payroll", "document_processing"]
        }
    )


@app.post("/process-document", response_model=DocumentProcessingResponse)
async def process_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    """
    Process document and extract text content only.
    Supports PDF, images, CSV, and Excel files.
    """
    processing_id = generate_processing_id()
    start_time = time.time()

    try:
        # Validate file
        validate_file(file)

        # Check file size
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > APIConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {APIConfig.MAX_FILE_SIZE_MB}MB"
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Track job
        track_job(processing_id, {
            'type': 'document_processing',
            'filename': file.filename,
            'file_size': file_size
        })

        logger.info(f"[{processing_id}] Processing document: {file.filename} ({file_size:,} bytes)")

        # Process document
        content = await process_document_content(file_content, file.filename)

        # Complete job tracking
        complete_job(processing_id, True)

        processing_time = time.time() - start_time

        logger.info(f"[{processing_id}] ✅ Document processed successfully in {processing_time:.2f}s")

        return DocumentProcessingResponse(
            success=True,
            message="Document processed successfully",
            processing_id=processing_id,
            filename=file.filename,
            file_type=get_file_type(file.filename),
            file_size=file_size,
            content_length=len(content),
            processing_time=processing_time,
            data={
                "content": content,
                "metadata": {
                    "processing_method": "ocr" if get_file_type(file.filename) in ['pdf', 'image'] else "direct",
                    "character_count": len(content)
                }
            }
        )

    except HTTPException:
        complete_job(processing_id, False)
        raise
    except Exception as e:
        complete_job(processing_id, False)
        logger.error(f"[{processing_id}] ❌ Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/extract-timesheet", response_model=TimesheetExtractionResponse)
async def extract_timesheet(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    """
    Process document and extract basic timesheet data.
    Returns 14 basic timesheet fields per employee record.
    """
    processing_id = generate_processing_id()
    start_time = time.time()

    try:
        # Validate file
        validate_file(file)

        # Check file size
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > APIConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {APIConfig.MAX_FILE_SIZE_MB}MB"
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Track job
        track_job(processing_id, {
            'type': 'timesheet_extraction',
            'filename': file.filename,
            'file_size': file_size
        })

        logger.info(f"[{processing_id}] Starting timesheet extraction: {file.filename} ({file_size:,} bytes)")

        # Step 1: Process document content
        logger.info(f"[{processing_id}] Step 1: Processing document content...")
        content = await process_document_content(file_content, file.filename)

        # Step 2: Find employees
        logger.info(f"[{processing_id}] Step 2: Finding employees...")
        employee_names = find_employees(content, APIConfig.ANTHROPIC_API_KEY)

        if not employee_names:
            logger.warning(f"[{processing_id}] No employees found in document")
            complete_job(processing_id, True)

            return TimesheetExtractionResponse(
                success=True,
                message="Document processed but no employees found",
                processing_id=processing_id,
                filename=file.filename,
                file_type=get_file_type(file.filename),
                employee_names=[],
                total_employees=0,
                total_records=0,
                employee_records=[],
                processing_time=time.time() - start_time
            )

        logger.info(f"[{processing_id}] Found {len(employee_names)} employees: {employee_names}")

        # Step 3: Extract timesheet records
        logger.info(f"[{processing_id}] Step 3: Extracting timesheet records...")
        client = anthropic.Anthropic(api_key=APIConfig.ANTHROPIC_API_KEY)

        all_records = []
        failed_employees = []

        for i, employee in enumerate(employee_names):
            try:
                logger.info(f"[{processing_id}] Processing employee {i + 1}/{len(employee_names)}: {employee}")

                records = extract_employee_records_with_continuation(
                    content, employee, client, i, len(employee_names)
                )

                if records:
                    all_records.extend(records)
                    logger.info(f"[{processing_id}] ✅ Extracted {len(records)} records for {employee}")
                else:
                    logger.warning(f"[{processing_id}] ⚠️ No records found for {employee}")
                    failed_employees.append(employee)

            except Exception as e:
                logger.error(f"[{processing_id}] ❌ Failed to process {employee}: {str(e)}")
                failed_employees.append(employee)

        # Step 4: Calculate summary
        timesheet_summary = {
            "total_employees_processed": len(employee_names) - len(failed_employees),
            "total_employees_failed": len(failed_employees),
            "failed_employees": failed_employees,
            "total_standard_hours": sum(float(r.get('Total Standard Hours', 0)) for r in all_records),
            "total_overtime_hours": sum(float(r.get('Total Over Time Hours', 0)) for r in all_records),
            "records_per_employee": {
                employee: len([r for r in all_records if r.get('employee_name') == employee])
                for employee in employee_names
            },
            "extraction_type": "basic_timesheet",
            "fields_extracted": 14
        }

        # Complete job tracking
        complete_job(processing_id, True)

        processing_time = time.time() - start_time

        logger.info(f"[{processing_id}] ✅ Timesheet extraction completed successfully in {processing_time:.2f}s")
        logger.info(f"[{processing_id}] 📊 Results: {len(all_records)} records from {len(employee_names)} employees")

        return TimesheetExtractionResponse(
            success=True,
            message="Timesheet extraction completed successfully",
            processing_id=processing_id,
            filename=file.filename,
            file_type=get_file_type(file.filename),
            employee_names=employee_names,
            total_employees=len(employee_names),
            total_records=len(all_records),
            employee_records=all_records,
            extraction_summary=timesheet_summary,
            processing_time=processing_time,
            data={
                "extraction_metadata": {
                    "content_length": len(content),
                    "processing_steps": 3,
                    "failed_employees": failed_employees,
                    "extraction_type": "basic_timesheet"
                }
            }
        )

    except HTTPException:
        complete_job(processing_id, False)
        raise
    except Exception as e:
        complete_job(processing_id, False)
        logger.error(f"[{processing_id}] ❌ Timesheet extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Timesheet extraction failed: {str(e)}")


@app.post("/extract-payroll", response_model=PayrollExtractionResponse)
async def extract_payroll(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...)
):
    """
    Process document and extract comprehensive payroll data.
    Returns structured payroll information with 28+ fields per employee.
    """
    processing_id = generate_processing_id()
    start_time = time.time()

    try:
        # Validate file
        validate_file(file)

        # Check file size
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > APIConfig.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {APIConfig.MAX_FILE_SIZE_MB}MB"
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Track job
        track_job(processing_id, {
            'type': 'payroll_extraction',
            'filename': file.filename,
            'file_size': file_size
        })

        logger.info(f"[{processing_id}] Starting payroll extraction: {file.filename} ({file_size:,} bytes)")

        # Step 1: Process document content
        logger.info(f"[{processing_id}] Step 1: Processing document content...")
        content = await process_document_content(file_content, file.filename)

        # Step 2: Find employees
        logger.info(f"[{processing_id}] Step 2: Finding employees...")
        employee_names = find_employees_enhanced(content, APIConfig.ANTHROPIC_API_KEY)

        if not employee_names:
            logger.warning(f"[{processing_id}] No employees found in document")
            complete_job(processing_id, True)

            return PayrollExtractionResponse(
                success=True,
                message="Document processed but no employees found",
                processing_id=processing_id,
                filename=file.filename,
                file_type=get_file_type(file.filename),
                employee_names=[],
                total_employees=0,
                total_records=0,
                employee_records=[],
                processing_time=time.time() - start_time
            )

        logger.info(f"[{processing_id}] Found {len(employee_names)} employees: {employee_names}")

        # Step 3: Extract payroll records
        logger.info(f"[{processing_id}] Step 3: Extracting payroll records...")
        client = anthropic.Anthropic(api_key=APIConfig.ANTHROPIC_API_KEY)

        all_records = []
        failed_employees = []

        for i, employee in enumerate(employee_names):
            try:
                logger.info(f"[{processing_id}] Processing employee {i + 1}/{len(employee_names)}: {employee}")

                records = extract_enhanced_payroll_records(
                    content, employee, client, i, len(employee_names)
                )

                if records:
                    # Calculate derived fields
                    for record in records:
                        try:
                            # Ensure numeric fields
                            regular_rate = float(record.get('regular_pay_rate', 0))
                            regular_hours = float(record.get('regular_hours', 0))
                            overtime_rate = float(record.get('overtime_pay_rate', 0))
                            overtime_hours = float(record.get('overtime_hours', 0))

                            # Calculate totals
                            record['regular_total'] = regular_rate * regular_hours
                            record['over_time_total'] = overtime_rate * overtime_hours
                            record['total_hours'] = regular_hours + overtime_hours

                            # Calculate pay totals
                            gross_amount = float(record.get('gross_amount_earned', 0))
                            per_diem = float(record.get('per_diem', 0))
                            record['total_pay'] = gross_amount + per_diem

                            # Calculate deduction totals
                            fica = float(record.get('fica', 0))
                            fed_tax = float(record.get('fed_wh_tax', 0))
                            state_tax = float(record.get('state_wh_tax', 0))
                            other_ded1 = float(record.get('other_deduction_1', 0))
                            other_ded2 = float(record.get('other_deduction_2', 0))
                            record['total_deductions'] = fica + fed_tax + state_tax + other_ded1 + other_ded2

                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"[{processing_id}] Error calculating totals for {record.get('record_id', 'unknown')}: {e}")

                    all_records.extend(records)
                    logger.info(f"[{processing_id}] ✅ Extracted {len(records)} records for {employee}")
                else:
                    logger.warning(f"[{processing_id}] ⚠️ No records found for {employee}")
                    failed_employees.append(employee)

            except Exception as e:
                logger.error(f"[{processing_id}] ❌ Failed to process {employee}: {str(e)}")
                failed_employees.append(employee)

        # Step 4: Calculate payroll summary
        payroll_summary = {
            "total_employees_processed": len(employee_names) - len(failed_employees),
            "total_employees_failed": len(failed_employees),
            "failed_employees": failed_employees,
            "total_regular_hours": sum(float(r.get('regular_hours', 0)) for r in all_records),
            "total_overtime_hours": sum(float(r.get('overtime_hours', 0)) for r in all_records),
            "total_gross_amount": sum(float(r.get('gross_amount_earned', 0)) for r in all_records),
            "total_per_diem": sum(float(r.get('per_diem', 0)) for r in all_records),
            "total_deductions": sum(float(r.get('total_deductions', 0)) for r in all_records),
            "total_net_wages": sum(float(r.get('net_wage_paid_for_week', 0)) for r in all_records),
            "average_hourly_rate": sum(float(r.get('regular_pay_rate', 0)) for r in all_records) / max(len(all_records),
                                                                                                       1),
            "records_per_employee": {
                employee: len([r for r in all_records if r.get('employee_name') == employee])
                for employee in employee_names
            },
            "extraction_type": "enhanced_payroll",
            "fields_extracted": 28
        }

        # Complete job tracking
        complete_job(processing_id, True)

        processing_time = time.time() - start_time

        logger.info(f"[{processing_id}] ✅ Payroll extraction completed successfully in {processing_time:.2f}s")
        logger.info(f"[{processing_id}] 📊 Results: {len(all_records)} records from {len(employee_names)} employees")

        return PayrollExtractionResponse(
            success=True,
            message="Payroll extraction completed successfully",
            processing_id=processing_id,
            filename=file.filename,
            file_type=get_file_type(file.filename),
            employee_names=employee_names,
            total_employees=len(employee_names),
            total_records=len(all_records),
            employee_records=all_records,
            payroll_summary=payroll_summary,
            processing_time=processing_time,
            data={
                "extraction_metadata": {
                    "content_length": len(content),
                    "processing_steps": 3,
                    "failed_employees": failed_employees,
                    "extraction_type": "enhanced_payroll"
                }
            }
        )

    except HTTPException:
        complete_job(processing_id, False)
        raise
    except Exception as e:
        complete_job(processing_id, False)
        logger.error(f"[{processing_id}] ❌ Payroll extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Payroll extraction failed: {str(e)}")


@app.get("/job-status/{processing_id}")
async def get_job_status(processing_id: str):
    """Get status of a processing job"""
    if processing_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_jobs[processing_id]

    status_info = {
        "processing_id": processing_id,
        "status": job['status'],
        "type": job['type'],
        "filename": job['filename'],
        "start_time": job['start_time'],
        "elapsed_time": time.time() - job['start_time']
    }

    if 'end_time' in job:
        status_info['end_time'] = job['end_time']
        status_info['total_time'] = job['end_time'] - job['start_time']

    return status_info


@app.get("/stats")
async def get_processing_stats():
    """Get processing statistics"""
    if ocr_processor:
        ocr_stats = ocr_processor.get_stats()
    else:
        ocr_stats = {}

    return {
        "active_jobs": len(active_jobs),
        "job_history": list(active_jobs.values())[-10:],  # Last 10 jobs
        "ocr_stats": ocr_stats,
        "api_config": {
            "max_file_size_mb": APIConfig.MAX_FILE_SIZE_MB,
            "max_concurrent_jobs": APIConfig.MAX_CONCURRENT_JOBS,
            "request_timeout": APIConfig.REQUEST_TIMEOUT
        },
        "extraction_capabilities": {
            "timesheet_fields": 14,
            "payroll_fields": 28,
            "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "CSV", "XLSX", "XLS"]
        }
    }


# ===========================
# ERROR HANDLERS
# ===========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "status_code": 500
        }
    )


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    # Load environment variables
    port = int(os.getenv('PORT', '8000'))
    host = os.getenv('HOST', '0.0.0.0')

    print("🚀 Starting Complete Document Processing API Server...")
    print(f"📍 Server will run on: http://{host}:{port}")
    print("🌐 Frontend UI: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Available Endpoints:")
    print("  • GET / - Frontend UI")
    print("  • POST /extract-timesheet - Basic timesheet extraction (14 fields)")
    print("  • POST /extract-payroll - Enhanced payroll extraction (28+ fields)")
    print("  • POST /process-document - Document text extraction only")
    print("  • GET /health - Health check")
    print("  • GET /stats - Processing statistics")
    print("💡 Make sure to set ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables")

    # Run server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Set to False for production
        access_log=True,
        log_level="info"
    )