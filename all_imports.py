import base64
import time, sys
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
import openai
import fitz
from PIL import Image
from io import BytesIO, StringIO
import pandas as pd
from dotenv import load_dotenv
import locale

# Configure structured logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Set console encoding to UTF-8 for Windows


if sys.platform.startswith('win'):
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Try to set console to UTF-8
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass


logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class ProcessingResult:
    """Structured result for processing operations"""
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    pages_processed: int = 0
    processing_time: float = 0.0
    file_size: int = 0
    metadata: Dict[str, Any] = None


class ProcessingConfig:
    """Configuration for OCR processing"""
    MAX_FILE_SIZE_MB = 100
    MAX_PAGES = 50
    DPI = 300
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    RATE_LIMIT_DELAY = 1
    REQUEST_TIMEOUT = 120
    TEMP_DIR = None  # Use system temp

    @classmethod
    def from_env(cls):
        """Load config from environment variables"""
        config = cls()
        config.MAX_FILE_SIZE_MB = int(os.getenv('OCR_MAX_FILE_SIZE_MB', '100'))
        config.MAX_PAGES = int(os.getenv('OCR_MAX_PAGES', '50'))
        config.DPI = int(os.getenv('OCR_DPI', '300'))
        config.MAX_RETRIES = int(os.getenv('OCR_MAX_RETRIES', '3'))
        config.REQUEST_TIMEOUT = int(os.getenv('OCR_REQUEST_TIMEOUT', '120'))
        return config


class EnhancedOCRProcessor:
    """Production-ready OCR processor using PyMuPDF (no Poppler dependency)"""

    def __init__(self, api_key: str, config: ProcessingConfig = None):
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.api_key = api_key
        self.config = config or ProcessingConfig.from_env()
        self.client = openai.OpenAI(api_key=api_key)
        self._validate_api_connection()

        # Processing stats
        self.stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_pages': 0,
            'total_processing_time': 0.0
        }

    def _validate_api_connection(self):
        """Validate API connection with proper error handling"""
        try:
            response = self.client.models.list()
            logger.info("[SUCCESS] OpenAI API connection validated")
            return True
        except openai.AuthenticationError:
            logger.error("[ERROR] Invalid OpenAI API key")
            raise ValueError("Invalid OpenAI API key")
        except openai.APIConnectionError:
            logger.error("[ERROR] Cannot connect to OpenAI API")
            raise ConnectionError("Cannot connect to OpenAI API")
        except Exception as e:
            logger.error(f"[ERROR] API validation failed: {e}")
            raise ConnectionError(f"API validation failed: {e}")

    def process_pdf_document(self, file_content: bytes, filename: str) -> ProcessingResult:
        """
        Process PDF document using PyMuPDF with comprehensive error handling
        """
        start_time = time.time()
        processing_id = str(uuid.uuid4())[:8]

        logger.info(f"[{processing_id}] Starting PDF processing: {filename}")

        try:
            # Validate input
            validation_result = self._validate_pdf_input(file_content, filename)
            if not validation_result.success:
                return validation_result

            # Convert PDF to images using PyMuPDF
            images = self._pdf_to_images_pymupdf(file_content, processing_id)
            if not images:
                return ProcessingResult(
                    success=False,
                    error="Failed to convert PDF to images",
                    file_size=len(file_content)
                )

            # Check page limit
            if len(images) > self.config.MAX_PAGES:
                logger.warning(f"[{processing_id}] PDF has {len(images)} pages, limiting to {self.config.MAX_PAGES}")
                images = images[:self.config.MAX_PAGES]

            # Process pages
            extracted_pages = []
            for i, image in enumerate(images, 1):
                logger.info(f"[{processing_id}] Processing page {i}/{len(images)}")

                try:
                    encoded_image = self._image_to_base64(image)
                    page_text = self._extract_text_with_retry(encoded_image, i, processing_id)
                    extracted_pages.append(page_text)

                    # Rate limiting
                    time.sleep(self.config.RATE_LIMIT_DELAY)

                except Exception as e:
                    logger.error(f"[{processing_id}] Failed to process page {i}: {e}")
                    extracted_pages.append(f"ERROR: Failed to process page {i}: {str(e)}")

            # Compile results
            result_text = "\n\n".join(extracted_pages)
            processing_time = time.time() - start_time

            # Update stats
            self.stats['total_files'] += 1
            self.stats['successful_files'] += 1
            self.stats['total_pages'] += len(images)
            self.stats['total_processing_time'] += processing_time

            logger.info(
                f"[{processing_id}] [SUCCESS] Processing complete: {len(result_text):,} characters in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                content=result_text,
                pages_processed=len(images),
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename,
                    'pages': len(images)
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['total_files'] += 1
            self.stats['failed_files'] += 1

            logger.error(f"[{processing_id}] [ERROR] Processing failed: {str(e)}")

            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename
                }
            )

    def process_csv_document(self, file_content: bytes, filename: str) -> ProcessingResult:
        """Process CSV file and return content as structured result"""
        start_time = time.time()
        processing_id = str(uuid.uuid4())[:8]

        logger.info(f"[{processing_id}] Starting CSV processing: {filename}")

        try:
            # Validate input
            if len(file_content) == 0:
                return ProcessingResult(success=False, error="Empty CSV file")

            if len(file_content) > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
                return ProcessingResult(success=False, error=f"CSV file too large (>{self.config.MAX_FILE_SIZE_MB}MB)")

            # Decode file content
            try:
                file_string = file_content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_string = file_content.decode("latin-1")
                    logger.info(f"[{processing_id}] Used latin-1 encoding for CSV")
                except UnicodeDecodeError:
                    return ProcessingResult(success=False, error="Unable to decode CSV file")

            # Read CSV
            df = pd.read_csv(StringIO(file_string))

            # Convert to string representation
            result_text = df.to_string()
            processing_time = time.time() - start_time

            # Update stats
            self.stats['total_files'] += 1
            self.stats['successful_files'] += 1
            self.stats['total_processing_time'] += processing_time

            logger.info(
                f"[{processing_id}] [SUCCESS] CSV processing complete: {len(df)} rows, {len(df.columns)} columns in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                content=result_text,
                pages_processed=1,
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'file_type': 'csv'
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['total_files'] += 1
            self.stats['failed_files'] += 1

            logger.error(f"[{processing_id}] [ERROR] CSV processing failed: {str(e)}")

            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename,
                    'file_type': 'csv'
                }
            )

    def process_excel_document(self, file_content: bytes, filename: str) -> ProcessingResult:
        """Process Excel file and return content as structured result with improved error handling"""
        start_time = time.time()
        processing_id = str(uuid.uuid4())[:8]

        logger.info(f"[{processing_id}] Starting Excel processing: {filename}")

        try:
            # Validate input
            if len(file_content) == 0:
                return ProcessingResult(success=False, error="Empty Excel file")

            if len(file_content) > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
                return ProcessingResult(success=False,
                                        error=f"Excel file too large (>{self.config.MAX_FILE_SIZE_MB}MB)")

            # Create BytesIO object from content
            file_obj = BytesIO(file_content)
            filename_lower = filename.lower()

            # Try different approaches based on file extension
            df = None
            engine_used = "unknown"

            if filename_lower.endswith('.xlsx'):
                try:
                    logger.info(f"[{processing_id}] Trying openpyxl engine for .xlsx file")
                    df = pd.read_excel(file_obj, engine='openpyxl')
                    engine_used = "openpyxl"
                except Exception as e:
                    logger.warning(f"[{processing_id}] openpyxl failed: {str(e)}, trying default engine")
                    file_obj.seek(0)
                    df = pd.read_excel(file_obj)
                    engine_used = "default"

            elif filename_lower.endswith('.xls'):
                try:
                    logger.info(f"[{processing_id}] Trying xlrd engine for .xls file")
                    df = pd.read_excel(file_obj, engine='xlrd')
                    engine_used = "xlrd"
                except Exception as e:
                    logger.warning(f"[{processing_id}] xlrd failed: {str(e)}, trying default engine")
                    file_obj.seek(0)
                    df = pd.read_excel(file_obj)
                    engine_used = "default"
            else:
                # Default approach for unknown extensions
                logger.info(f"[{processing_id}] Using default engine for Excel file")
                df = pd.read_excel(file_obj)
                engine_used = "default"

            # Convert to string representation
            result_text = df.to_string()
            processing_time = time.time() - start_time

            # Update stats
            self.stats['total_files'] += 1
            self.stats['successful_files'] += 1
            self.stats['total_processing_time'] += processing_time

            logger.info(
                f"[{processing_id}] [SUCCESS] Excel processing complete with {engine_used}: {len(df)} rows, {len(df.columns)} columns in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                content=result_text,
                pages_processed=1,
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'engine_used': engine_used,
                    'file_type': 'excel'
                }
            )

        except Exception as e:
            # Try one more time with a different approach
            try:
                logger.info(f"[{processing_id}] Attempting final fallback method")
                file_obj = BytesIO(file_content)
                df = pd.read_excel(file_obj, engine=None)  # Let pandas decide
                engine_used = "fallback"

                result_text = df.to_string()
                processing_time = time.time() - start_time

                # Update stats
                self.stats['total_files'] += 1
                self.stats['successful_files'] += 1
                self.stats['total_processing_time'] += processing_time

                logger.info(
                    f"[{processing_id}] [SUCCESS] Excel processed with fallback method: {len(df)} rows, {len(df.columns)} columns in {processing_time:.2f}s")

                return ProcessingResult(
                    success=True,
                    content=result_text,
                    pages_processed=1,
                    processing_time=processing_time,
                    file_size=len(file_content),
                    metadata={
                        'processing_id': processing_id,
                        'filename': filename,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'engine_used': engine_used,
                        'file_type': 'excel'
                    }
                )

            except Exception as fallback_error:
                processing_time = time.time() - start_time
                self.stats['total_files'] += 1
                self.stats['failed_files'] += 1

                logger.error(
                    f"[{processing_id}] [ERROR] Excel processing failed, fallback also failed: {str(fallback_error)}")
                logger.error(f"[{processing_id}] Original error: {str(e)}")

                return ProcessingResult(
                    success=False,
                    error=f"Excel processing failed: {str(e)}. Fallback error: {str(fallback_error)}",
                    processing_time=processing_time,
                    file_size=len(file_content),
                    metadata={
                        'processing_id': processing_id,
                        'filename': filename,
                        'file_type': 'excel'
                    }
                )
        """Process image document with comprehensive error handling"""
        start_time = time.time()
        processing_id = str(uuid.uuid4())[:8]

        logger.info(f"[{processing_id}] Starting image processing: {filename}")

        try:
            # Validate input
            if len(file_content) == 0:
                return ProcessingResult(success=False, error="Empty image file")

            if len(file_content) > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
                return ProcessingResult(success=False,
                                        error=f"Image file too large (>{self.config.MAX_FILE_SIZE_MB}MB)")

            # Process image
            image = Image.open(BytesIO(file_content))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            encoded_image = self._image_to_base64(image)
            result_text = self._extract_text_with_retry(encoded_image, 1, processing_id)

            processing_time = time.time() - start_time

            # Update stats
            self.stats['total_files'] += 1
            self.stats['successful_files'] += 1
            self.stats['total_pages'] += 1
            self.stats['total_processing_time'] += processing_time

            logger.info(
                f"[{processing_id}] [SUCCESS] Image processing complete: {len(result_text):,} characters in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                content=result_text,
                pages_processed=1,
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['total_files'] += 1
            self.stats['failed_files'] += 1

            logger.error(f"[{processing_id}] [ERROR] Image processing failed: {str(e)}")

            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename
                }
            )



    def process_image_document(self, file_content: bytes, filename: str) -> ProcessingResult:
        """Process image document with comprehensive error handling"""
        start_time = time.time()
        processing_id = str(uuid.uuid4())[:8]

        logger.info(f"[{processing_id}] Starting image processing: {filename}")

        try:
            # Validate input
            if len(file_content) == 0:
                return ProcessingResult(success=False, error="Empty image file")

            if len(file_content) > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
                return ProcessingResult(success=False,
                                        error=f"Image file too large (>{self.config.MAX_FILE_SIZE_MB}MB)")

            # Process image
            image = Image.open(BytesIO(file_content))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            encoded_image = self._image_to_base64(image)
            result_text = self._extract_text_with_retry(encoded_image, 1, processing_id)

            processing_time = time.time() - start_time

            # Update stats
            self.stats['total_files'] += 1
            self.stats['successful_files'] += 1
            self.stats['total_pages'] += 1
            self.stats['total_processing_time'] += processing_time

            logger.info(
                f"[{processing_id}] [SUCCESS] Image processing complete: {len(result_text):,} characters in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                content=result_text,
                pages_processed=1,
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename,
                    'image_size': f"{image.size[0]}x{image.size[1]}",
                    'image_mode': image.mode,
                    'file_type': 'image'
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['total_files'] += 1
            self.stats['failed_files'] += 1

            logger.error(f"[{processing_id}] [ERROR] Image processing failed: {str(e)}")

            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                file_size=len(file_content),
                metadata={
                    'processing_id': processing_id,
                    'filename': filename,
                    'file_type': 'image'
                }
            )

    def _validate_pdf_input(self, file_content: bytes, filename: str) -> ProcessingResult:
        """Validate PDF input with security checks"""
        if len(file_content) == 0:
            return ProcessingResult(success=False, error="Empty PDF file")

        if len(file_content) > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
            return ProcessingResult(success=False, error=f"PDF file too large (>{self.config.MAX_FILE_SIZE_MB}MB)")

        # Basic PDF header validation
        if not file_content.startswith(b'%PDF-'):
            return ProcessingResult(success=False, error="Invalid PDF file format")

        # Filename validation
        if not filename or len(filename) > 255:
            return ProcessingResult(success=False, error="Invalid filename")

        return ProcessingResult(success=True)

    def _pdf_to_images_pymupdf(self, file_content: bytes, processing_id: str) -> List[Image.Image]:
        """Convert PDF to images using PyMuPDF (no Poppler dependency)"""
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=file_content, filetype="pdf")

            total_pages = len(pdf_document)
            max_pages = min(total_pages, self.config.MAX_PAGES)

            logger.info(f"[{processing_id}] PDF has {total_pages} pages, processing {max_pages}")

            images = []

            for page_num in range(max_pages):
                try:
                    # Get the page
                    page = pdf_document[page_num]

                    # Create transformation matrix for DPI
                    zoom = self.config.DPI / 72.0  # 72 is default DPI
                    mat = fitz.Matrix(zoom, zoom)

                    # Render page to pixmap
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(BytesIO(img_data))

                    images.append(image)

                except Exception as e:
                    logger.error(f"[{processing_id}] Error processing page {page_num + 1}: {e}")
                    continue

            # Close the PDF document
            pdf_document.close()

            logger.info(f"[{processing_id}] Successfully converted {len(images)} pages to images")
            return images

        except Exception as e:
            logger.error(f"[{processing_id}] Error converting PDF to images with PyMuPDF: {e}")
            return []

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 with optimization"""
        try:
            # Optimize image size for API
            max_size = (2048, 2048)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

            buffered = BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise

    def _extract_text_with_retry(self, base64_image: str, page_num: int, processing_id: str) -> str:
        """Extract text with exponential backoff and proper error handling"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                logger.debug(f"[{processing_id}] Processing page {page_num}, attempt {attempt + 1}")

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Extract all text from this image exactly as it appears. 
                                    Preserve formatting, spacing, and layout. 
                                    If it's a table, maintain the table structure.
                                    Do not add or remove any content."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=16000,
                    timeout=self.config.REQUEST_TIMEOUT
                )

                text = response.choices[0].message.content

                # Validation
                if text and len(text.strip()) > 5 and not text.lower().startswith(('i cannot', 'unable to', 'sorry')):
                    logger.debug(f"[{processing_id}] [SUCCESS] Successfully extracted text from page {page_num}")
                    return text
                else:
                    logger.warning(f"[{processing_id}] Got empty/invalid response for page {page_num}, retrying...")

            except openai.RateLimitError as e:
                logger.warning(f"[{processing_id}] Rate limit hit for page {page_num}, waiting...")
                time.sleep(min(60, 5 * (2 ** attempt)))
            except openai.APITimeoutError as e:
                logger.warning(f"[{processing_id}] Timeout for page {page_num}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"[{processing_id}] Attempt {attempt + 1} failed for page {page_num}: {e}")

            # Exponential backoff
            if attempt < self.config.MAX_RETRIES - 1:
                delay = self.config.RETRY_DELAY * (2 ** attempt)
                time.sleep(delay)

        error_msg = f"Failed to extract text from page {page_num} after {self.config.MAX_RETRIES} attempts"
        logger.error(f"[{processing_id}] [ERROR] {error_msg}")
        return f"ERROR: {error_msg}"

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'success_rate': self.stats['successful_files'] / max(self.stats['total_files'], 1) * 100,
            'avg_processing_time': self.stats['total_processing_time'] / max(self.stats['total_files'], 1),
            'avg_pages_per_file': self.stats['total_pages'] / max(self.stats['successful_files'], 1)
        }

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_pages': 0,
            'total_processing_time': 0.0
        }





# # Example usage
# if __name__ == "__main__":
#     # Load configuration
#     config = ProcessingConfig.from_env()
#     api_key = os.getenv('OPENAI_API_KEY')
#     file_path = r"C:\Users\GAYATHRI K\OneDrive\Desktop\universal_import_tim_pay\universal_import_tim_pay\Richard Baughn Payroll Example - QB (1).png"
#     processor = EnhancedOCRProcessor(api_key, config)
#     with open(file_path, 'rb') as f:
#         file_content = f.read()
#     result = processor.process_image_document(file_content, "Richard Baughn Payroll Example - QB (1).png")
#     print(result.content)