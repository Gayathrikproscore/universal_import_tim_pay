import json
import anthropic
import re
import logging
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
import time
from functools import wraps


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimesheetProcessingError(Exception):
    """Custom exception for timesheet processing errors"""
    pass


def retry_on_failure(max_retries: int = 3):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except anthropic.RateLimitError as e:
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise
            return None
        return wrapper
    return decorator


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from Claude's response text."""
    if not response_text:
        raise ValueError("Empty response text")

    # Try to find JSON in code blocks first
    if '```json' in response_text:
        start = response_text.find('```json') + 7
        end = response_text.find('```', start)
        if end != -1:
            json_candidate = response_text[start:end].strip()
            if json_candidate.startswith('{'):
                return json_candidate

    # Fallback to finding JSON boundaries
    start = response_text.find('{')
    end = response_text.rfind('}') + 1

    if start == -1 or end <= start:
        raise ValueError("No valid JSON found in response")

    json_text = response_text[start:end]

    # Fix common JSON issues
    if not json_text.strip().endswith('}'):
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        missing_braces = open_braces - close_braces
        if missing_braces > 0:
            json_text += '}' * missing_braces

    # Remove trailing commas
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

    return json_text


@retry_on_failure(max_retries=3)
def find_employees(file_content: str, api_key: str) -> List[str]:
    """Find all employee names with full dataset visibility"""
    if not file_content or not file_content.strip():
        raise ValueError("File content is empty")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Analyze this complete timesheet dataset to find ALL unique employee names.

COMPLETE DATASET:
{file_content}

INSTRUCTIONS:
1. Look through EVERY column and EVERY row
2. Employee names can appear in any column (Name, Employee, Worker, etc.)
3. Look for patterns like "FirstName LastName" or single names
4. Include variations and nicknames if they refer to the same person
5. Be thorough - this is the foundation for all subsequent processing
6. Return only actual names, not headers or column names

Look in ALL columns and ALL rows. Employee names can be in any column.
Search thoroughly through the entire dataset.

Return ONLY a JSON object:
{{
    "employee_names": ["Name1", "Name2", "Name3", ...]
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0,
            system="You are an expert at analyzing timesheet data. Find ALL unique employee names by examining every part of the dataset thoroughly.",
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text.strip()

        try:
            json_text = extract_json_from_response(response_text)
            result = json.loads(json_text)
            employee_names = result.get("employee_names", [])

            # Filter out empty names
            employee_names = [name.strip() for name in employee_names if name and name.strip()]

            if not employee_names:
                logger.warning("No employee names found in the dataset")
                return []

            logger.info(f"Found {len(employee_names)} employees: {employee_names}")
            return employee_names

        except Exception as e:
            logger.error(f"Error parsing employee names response: {str(e)}")
            return []

    except Exception as e:
        logger.error(f"Error finding employees: {str(e)}")
        raise TimesheetProcessingError(f"Failed to find employees: {str(e)}")


@retry_on_failure(max_retries=3)
def extract_employee_records_with_continuation(
    full_data: str,
    employee: str,
    client: anthropic.Anthropic,
    emp_index: int,
    total_employees: int
) -> List[Dict[str, Any]]:
    """Extract all records for a specific employee with continuation handling"""

    initial_prompt = f"""Extract ALL timesheet records for employee: "{employee}"

CRITICAL CONTEXT:
- You have the COMPLETE dataset below
- Pay close attention to every field and record with Eagle-eye precision.
- This employee may have MULTIPLE records/entries
- Under no circumstances should any 'Single' employee-related records be overlooked. 
- Each Single record are critical—exercise utmost diligence and ensure every detail is thoroughly reviewed and captured without exception.
- This timesheet may contain MULTIPLE ENTRIES for the same employee on the same work date. This is NORMAL and EXPECTED.
- Double-check thoroughly to ensure that no employee records are missing from the input data. 
- Every Single entry must be accounted for without exception.
- Each row or entry should be treated as a SEPARATE RECORD, even if it's the same employee and same date.
- Each ROW in the data = ONE SEPARATE RECORD (even if same employee + same date)
- Multiple entries for same employee on same date are EXPECTED and CORRECT, and treated as SEPARATE RECORD
- Extract EVERY occurrence of this employee even if its same date.
- If there are no multiple entries and the row reference (indicating the source row or line of the data) is the same, please avoid duplicating the records.

COMPLETE DATASET:
{full_data}

EXTRACTION RULES:
1. Find EVERY row/entry containing "{employee}"
2. Each row = one separate record (even if same date)
3. Create unique record_id: "{employee}_Date_RowNum"
4. Double-check thoroughly to ensure that No single employee records are missing from the input data. Every Single entry must be accounted for without exception.
5. Extract all available information per record
6. Use " " for missing text, 0 for missing numbers

For EACH individual record/entry, extract the following information:
    1. Record ID (create a unique identifier like "EmployeeName_Date_SequenceNumber")
    2. Full employee name (as it appears in the data)
    3. First name and last name (split the full name)
    4. Email address (if available)
    5. Work date (in MM-DD-YYYY format)
    6. Job position/title (Any job related information)
    7. Standard pay rate (IMPORTANT: Extract the numeric value for THIS specific record)
    8. Total standard working hours (numeric value for THIS specific record)
    9. Total Overtime pay rate/ OT pay rate / OV pay rate (numeric value for THIS specific record): Overtime pay rate
    10. Total Overtime hours/ OT hours/ OV hours (numeric value for THIS specific record): Overtime hours
    11. Per diem rate (numeric value for THIS specific record)
    12. Week ending date (in MM-DD-YYYY format)
    13. Employee ID/ ID (as it appears in the data)
    14. Row reference (indicate which row/line this data came from)

    Strictly ensure that all the above-mentioned information is extracted from the data if present. Do not ignore or omit any detail under any circumstances.

    CRITICAL INSTRUCTIONS:
    - Treat EVERY row as a separate record, even if employee and work date are the same
    - Do NOT combine or aggregate records with the same employee and work date
    - If any information is not available, use " " for text fields and 0 for numeric fields
    - Create unique record IDs to distinguish between multiple entries
    - Extract EVERY record you can find, maintaining all separate entries
    - If there are no multiple entries and the row reference (indicating the source row or line of the data) is the same, please avoid duplicating the records.
    - Double-check thoroughly to ensure that no employee records are missing from the input data. Every single entry must be accounted for without exception.
    - Do not hallucinate or assume any values or records. Only extract and process information that is explicitly present in the data.

REQUIRED FIELDS per record:
- record_id, employee_name, First Name, Last Name,
- Email, Worked Date (MM-DD-YYYY), Position,
- Standard Pay Rate, Total Standard Hours,
- Over Time Pay Rate, Total Over Time Hours,
- Per Diem, Week Ending Date, Employee Internal Id, row_reference

Return JSON:
{{
    "employee_records": [
        {{
            "record_id": "string",
            "employee_name": "{employee}",
            "First Name": "string",
            "Last Name": "string", 
            "Email": "string",
            "Worked Date": "MM-DD-YYYY",
            "Position": "string",
            "Standard Pay Rate": 0,
            "Total Standard Hours": 0,
            "Over Time Pay Rate": 0,
            "Total Over Time Hours": 0,
            "Per Diem": 0,
            "Week Ending Date": "MM-DD-YYYY",
            "Employee Internal Id": "string",
            "row_reference": "string"
        }}
    ],
    "extraction_complete": true,
    "records_found": 0
}}

Process systematically and thoroughly."""

    try:
        logger.info(f"Extracting records for employee: {employee} ({emp_index + 1}/{total_employees})")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0,
            system=f"Extract ALL records for the specified employee from the complete dataset. Employee {emp_index + 1} of {total_employees}. Be thorough and systematic.",
            messages=[{"role": "user", "content": initial_prompt}]
        )
        response_text = response.content[0].text.strip()

        # Handle max tokens reached
        if response.stop_reason == "max_tokens":
            logger.warning(f"Max tokens reached for {employee}, handling continuation...")
            return handle_employee_continuation(full_data, employee, client, response_text)

        try:
            json_text = extract_json_from_response(response_text)
            result = json.loads(json_text)
            employee_records = result.get("employee_records", [])
            extraction_complete = result.get("extraction_complete", True)

            if not extraction_complete:
                logger.warning(f"Extraction incomplete for {employee}, handling continuation...")
                return handle_employee_continuation(full_data, employee, client, response_text)

            logger.info(f"Successfully extracted {len(employee_records)} records for {employee}")
            return employee_records

        except Exception as e:
            logger.error(f"Error parsing JSON response for {employee}: {str(e)}")
            return salvage_partial_employee_records(response_text, employee)

    except Exception as e:
        logger.error(f"Error extracting employee records for {employee}: {str(e)}")
        return []


def handle_employee_continuation(
    full_data: str,
    employee: str,
    client: anthropic.Anthropic,
    partial_response: str
) -> List[Dict[str, Any]]:
    """Handle continuation for a specific employee's extraction"""

    partial_records = []
    try:
        partial_json = extract_json_from_response(partial_response)
        partial_result = json.loads(partial_json)
        partial_records = partial_result.get("employee_records", [])
        logger.info(f"Found {len(partial_records)} partial records for {employee}")
    except Exception as e:
        logger.error(f"Error parsing partial response for {employee}: {str(e)}")

    continuation_prompt = f"""CONTINUATION: Complete the extraction for employee "{employee}"
CONTEXT:
- This is a continuation of a previous extraction
- Already extracted {len(partial_records)} records (if any)
- Need to complete the full extraction for this employee
COMPLETE DATASET (same as before):
{full_data}
CONTINUATION TASK:
1. Review the complete dataset again for "{employee}"
2. Extract ALL remaining records not yet captured
3. Ensure no records are missed
4. Each row with this employee = separate record
5. Ensure and Maintain all same Instructions and JSON format as before

Previous partial results context: {len(partial_records)} records already found

Continue systematic extraction for "{employee}" and return:
{{
    "employee_records": [complete_list_of_all_records],
    "total_records": number,
    "continuation_complete": true
}}

Be thorough and complete the extraction."""

    try:
        logger.info(f"Continuing extraction for {employee}...")

        continuation_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0,
            system=f"Complete the extraction for {employee}. This is a continuation - be thorough and systematic.",
            messages=[{"role": "user", "content": continuation_prompt}]
        )
        continuation_text = continuation_response.content[0].text.strip()

        try:
            json_text = extract_json_from_response(continuation_text)
            result = json.loads(json_text)
            continued_records = result.get("employee_records", [])

            # Combine with partial records, avoiding duplicates
            all_employee_records = partial_records.copy()
            existing_record_ids = {record.get("record_id") for record in partial_records}

            for record in continued_records:
                if record.get("record_id") not in existing_record_ids:
                    all_employee_records.append(record)

            logger.info(f"Continuation complete for {employee}: {len(all_employee_records)} total records")
            return all_employee_records

        except Exception as e:
            logger.error(f"Error parsing continuation response for {employee}: {str(e)}")
            return partial_records

    except Exception as e:
        logger.error(f"Error in continuation for {employee}: {str(e)}")
        return partial_records


def salvage_partial_employee_records(response_text: str, employee: str) -> List[Dict[str, Any]]:
    """Attempt to salvage partial records from malformed response"""
    try:
        logger.info(f"Attempting to salvage partial records for {employee}")

        record_pattern = r'\{[^{}]*"record_id"[^{}]*"' + re.escape(employee) + r'"[^{}]*\}'
        potential_records = re.findall(record_pattern, response_text, re.IGNORECASE)
        salvaged_records = []

        for record_str in potential_records:
            try:
                record = json.loads(record_str)
                if "record_id" in record and employee.lower() in record.get("employee_name", "").lower():
                    salvaged_records.append(record)
            except:
                continue

        if salvaged_records:
            logger.info(f"Salvaged {len(salvaged_records)} records for {employee}")
            return salvaged_records
        else:
            logger.warning(f"No records could be salvaged for {employee}")
            return []

    except Exception as e:
        logger.error(f"Error salvaging records for {employee}: {str(e)}")
        return []


def process_timesheet(file_path: str, api_key: str) -> Dict[str, Any]:
    """Main function to process timesheet file"""
    try:
        # Validate inputs
        if not api_key:
            raise ValueError("API key is required")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        if not file_content.strip():
            raise ValueError("File is empty")

        logger.info(f"Processing timesheet file: {file_path}")

        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)

        # Find all employees
        employees = find_employees(file_content, api_key)
        if not employees:
            raise TimesheetProcessingError("No employees found in the timesheet")

        # Extract records for each employee
        all_records = []
        failed_employees = []

        for i, employee in enumerate(employees):
            try:
                employee_records = extract_employee_records_with_continuation(
                    file_content, employee, client, i, len(employees)
                )
                all_records.extend(employee_records)
            except Exception as e:
                logger.error(f"Failed to process employee {employee}: {str(e)}")
                failed_employees.append(employee)

        # Return results
        results = {
            "success": True,
            "total_employees": len(employees),
            "processed_employees": len(employees) - len(failed_employees),
            "failed_employees": failed_employees,
            "total_records": len(all_records),
            "records": all_records,
            "processing_timestamp": datetime.now().isoformat(),
            "file_processed": file_path
        }

        logger.info(f"Processing complete: {len(all_records)} records from {len(employees)} employees")
        return results

    except Exception as e:
        logger.error(f"Error processing timesheet: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "processing_timestamp": datetime.now().isoformat()
        }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


# def main():
#     """Simple command-line usage"""
#     import sys
#
#     if len(sys.argv) < 2:
#         print("Usage: python timesheet_processor.py <input_file> [output_file]")
#         print("Set ANTHROPIC_API_KEY environment variable")
#         sys.exit(1)
#
#     input_file = sys.argv[1]
#     output_file = sys.argv[2] if len(sys.argv) > 2 else "timesheet_results.json"
#
#     # Get API key from environment
#     api_key = os.getenv("ANTHROPIC_API_KEY")
#     if not api_key:
#         print("Error: ANTHROPIC_API_KEY environment variable not set")
#         sys.exit(1)
#
#     # Process timesheet
#     print(f"Processing {input_file}...")
#     results = process_timesheet(input_file, api_key)
#
#     # Save results
#     save_results(results, output_file)
#
#     # Print summary
#     if results["success"]:
#         print(f"✅ Success! Processed {results['processed_employees']}/{results['total_employees']} employees")
#         print(f"📝 Extracted {results['total_records']} records")
#         print(f"💾 Results saved to: {output_file}")
#     else:
#         print(f"❌ Failed: {results['error']}")
#
#
# if __name__ == "__main__":
#     main()