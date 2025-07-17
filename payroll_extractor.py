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


class PayrollProcessingError(Exception):
    """Custom exception for payroll processing errors"""
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
                        time.sleep(2 ** attempt)
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
def find_employees_enhanced(file_content: str, api_key: str) -> List[str]:
    """Find all employee names with full dataset visibility - Enhanced version"""
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
            max_tokens=8192,
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
        raise PayrollProcessingError(f"Failed to find employees: {str(e)}")


@retry_on_failure(max_retries=3)
def extract_enhanced_payroll_records(
        full_data: str,
        employee: str,
        client: anthropic.Anthropic,
        emp_index: int,
        total_employees: int
) -> List[Dict[str, Any]]:
    """Extract all records for a specific employee with enhanced payroll fields"""

    initial_prompt = f"""Extract ALL timesheet records for employee: "{employee}"

CRITICAL CONTEXT:
- You have the COMPLETE dataset below
- This employee may have MULTIPLE records/entries
- This timesheet may contain MULTIPLE ENTRIES for the same employee on the same work date. This is NORMAL and EXPECTED.
- Each Row or entry should be treated as a SEPARATE RECORD, even if it's the same employee and same date.
- Each ROW in the data = ONE SEPARATE RECORD (even if same employee + same date)
- Multiple entries for same employee on same date are EXPECTED and CORRECT, and treated as SEPARATE RECORD
- Extract EVERY occurrence of this employee even if its same date.
- Both Regular and Overtime information should be consolidated into a Single record per employee; do not separate them into individual entries..
- Double-check thoroughly to ensure that no Single employee records are missing from the input data. 
- Every Single entry must be accounted for without exception.
- If there are no multiple entries and the row reference (indicating the source row or line of the data) is the same, please avoid duplicating the records.
- If two or more records refer to the same row reference, please discard the duplicate entries.

COMPLETE DATASET:
{full_data}

EXTRACTION RULES:
1. Find EVERY row/entry containing "{employee}"
2. Each row = one separate record (even if same date)
3. Create unique record_id: "{employee}_Date_RowNum"
4. Extract all available information per record
5. Both Regular and Overtime information should be consolidated into a Single record per employee; do not separate them into individual entries..
6. Use " " for missing text, 0 for missing numbers

For EACH individual record/entry, extract the following fields exactly as specified:

REQUIRED FIELDS per record (extract exactly these 25+ fields):
1. Week Ending Date (in MM-DD-YYYY format)
2. Last Name
3. First Name  
4. Job Position: Job related information present in the document.
5. Regular Pay Rate (numeric value): Regular Pay Rate information present in the document
6. Regular Hours (numeric value): Regular Hours information present in the document
7. Over Time Pay Rate (numeric value): Over Time Pay Rate information present in the document
8. Over Time Hours (numeric value): Over Time Hours information present in the document
9. Per Diem (numeric value): Per Diem information present in the document
10. Gross Amount Earned (numeric value): Gross Amount information present in the document
11. FICA (numeric value): FICA information present in the document
12. FED WH TAX (numeric value): FED WH TAX information present in the document
13. STATE WH TAX (numeric value): STATE WH TAX information present in the document
14. Other Deduction 1 (numeric value): Other Deduction 1 information present in the document
15. Other Deduction 2 (numeric value): Other Deduction 2 information present in the document
16. Net Wage Paid For Week (numeric value): Net Wage Paid For Week information present in the document
17. Other Deduction 1 Explanation (text): Other Deduction 1 Explanation information present in the document
18. Other Deduction 2 Explanation (text): Other Deduction 2 Explanation information present in the document
19. Exception (text or Y/N): Exception information present in the document
20. Explanation (text): Explanation present in the document
21. Employee ID/ ID (as it appears in the data): Any Employee ID present in the document
22. Record ID (create unique identifier: "EmployeeName_Date_SequenceNumber")
23. Row Reference (indicate which row/line this data came from)

NOTE: The following fields will be calculated automatically after extraction:
24. Regular Total = Regular Pay Rate * Regular Hours (numeric value)
25. Over Time Total = Over Time Pay Rate * Over Time Hours  (numeric value) 
26. Total Hours = Regular Hours + Over Time Hours (numeric value)
27. Total Pay = Gross Amount Earned + Per Diem (numeric value)
28. Total Deductions = FICA + FED WH TAX + STATE WH TAX + Other Deduction 1 + Other Deduction 2 (numeric value)

Strictly ensure that all the above-mentioned information is extracted from the data if present. Do not ignore or omit any detail under any circumstances.

CRITICAL INSTRUCTIONS:
- Treat EVERY row as a separate record, even if employee and work date are the same
- Do NOT combine or aggregate records with the same employee and work date
- If any information is not available, use " " for text fields and 0 for numeric fields
- Create unique record IDs to distinguish between multiple entries
- Extract EVERY record you can find, maintaining all separate entries
- Look for tax and deduction information in any columns
- Search for pay rates, hours, and financial data across all columns
- If there are no multiple entries and the row reference (indicating the source row or line of the data) is the same, please avoid duplicating the records.
- If two or more records refer to the same row reference, please discard the duplicate entries.
- Do not hallucinate or assume any values, fields or records. Only extract and process information that is explicitly present in the data.

NEGATE INSTRUCTIONS:
- Regular and overtime information must never be treated as separate entries. Always consolidate them into a single record per employee, without exception—even inadvertently.
- If two or more records refer to the same row reference, please discard the duplicate entries.
- Do not hallucinate or assume any values, fields or records. Only extract and process information that is explicitly present in the data.

Return JSON:
{{
    "employee_records": [
        {{
            "record_id": "string",
            "employee_name": "{employee}",
            "week_ending_date": "MM-DD-YYYY",
            "Last Name": "string",
            "First Name": "string",
            "Job Position": "string",
            "Regular Pay Rate": 0,
            "Regular Hours": 0,
            "Regular Total": 0,
            "Over Time Pay Rate": 0,
            "Over Time Hours": 0,
            "Over Time Total": 0, 
            "Total Hours": 0,
            "Per Diem": 0,
            "Gross Amount Earned": 0,
            "Total Pay": 0,
            "FICA": 0,
            "FED WH TAX": 0,
            "STATE WH TAX": 0,
            "Other Deduction 1": 0,
            "Other Deduction 2": 0,
            "Net Wage Paid For Week": 0,
            "Total Deductions": 0,
            "Other Deduction 1 Explanation": "string",
            "Other Deduction 2 Explanation": "string",
            "Exception": "string",
            "Explanation": "string",
            "Employee Internal Id": "string",
            "row_reference": "string"
        }}
    ],
    "extraction_complete": true,
    "records_found": 0
}}

Process systematically and thoroughly."""

    try:
        logger.info(f"Extracting enhanced payroll records for employee: {employee} ({emp_index + 1}/{total_employees})")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0,
            system=f"Extract ALL records for the specified employee from the complete dataset. Employee {emp_index + 1} of {total_employees}. Be thorough and systematic. Extract all required payroll fields as specified.",
            messages=[{"role": "user", "content": initial_prompt}]
        )
        response_text = response.content[0].text.strip()

        # Handle max tokens reached
        if response.stop_reason == "max_tokens":
            logger.warning(f"Max tokens reached for {employee}, handling continuation...")
            return handle_payroll_continuation(full_data, employee, client, response_text)

        try:
            json_text = extract_json_from_response(response_text)
            result = json.loads(json_text)
            employee_records = result.get("employee_records", [])
            extraction_complete = result.get("extraction_complete", True)

            if not extraction_complete:
                logger.warning(f"Extraction incomplete for {employee}, handling continuation...")
                return handle_payroll_continuation(full_data, employee, client, response_text)

            logger.info(f"Successfully extracted {len(employee_records)} enhanced payroll records for {employee}")
            return employee_records

        except Exception as e:
            logger.error(f"Error parsing JSON response for {employee}: {str(e)}")
            return salvage_partial_payroll_records(response_text, employee)

    except Exception as e:
        logger.error(f"Error extracting enhanced payroll records for {employee}: {str(e)}")
        return []


def handle_payroll_continuation(
        full_data: str,
        employee: str,
        client: anthropic.Anthropic,
        partial_response: str
) -> List[Dict[str, Any]]:
    """Handle continuation for enhanced payroll extraction"""

    partial_records = []
    try:
        partial_json = extract_json_from_response(partial_response)
        partial_result = json.loads(partial_json)
        partial_records = partial_result.get("employee_records", [])
        logger.info(f"Found {len(partial_records)} partial records for {employee}")
    except Exception as e:
        logger.error(f"Error parsing partial response for {employee}: {str(e)}")

    continuation_prompt = f"""CONTINUATION: Complete the enhanced payroll extraction for employee "{employee}"

CONTEXT:
- This is a continuation of a previous extraction
- Already extracted {len(partial_records)} records (if any)
- Need to complete the full extraction for this employee
- Must extract all required payroll fields as specified in pipeline

COMPLETE DATASET (same as before):
{full_data}

CONTINUATION TASK:
1. Review the complete dataset again for "{employee}"
2. Extract ALL remaining records not yet captured
3. Ensure no records are missed
4. Each row with this employee = separate record
5. Maintain all same field requirements and JSON format as before
6. Include all tax, deduction, and payroll fields

Previous partial results context: {len(partial_records)} records already found

Continue systematic extraction for "{employee}" and return:
{{
    "employee_records": [complete_list_of_all_records_with_all_required_fields],
    "total_records": number,
    "continuation_complete": true
}}

Be thorough and complete the extraction with all required payroll fields."""

    try:
        logger.info(f"Continuing enhanced payroll extraction for {employee}...")

        continuation_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            temperature=0,
            system=f"Complete the enhanced payroll extraction for {employee}. This is a continuation - be thorough and systematic. Extract all required payroll fields.",
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

            logger.info(
                f"Enhanced payroll continuation complete for {employee}: {len(all_employee_records)} total records")
            return all_employee_records

        except Exception as e:
            logger.error(f"Error parsing continuation response for {employee}: {str(e)}")
            return partial_records

    except Exception as e:
        logger.error(f"Error in enhanced payroll continuation for {employee}: {str(e)}")
        return partial_records


def salvage_partial_payroll_records(response_text: str, employee: str) -> List[Dict[str, Any]]:
    """Try to salvage partial enhanced payroll records from malformed response"""
    try:
        logger.info(f"Attempting to salvage partial enhanced payroll records for {employee}")

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
            logger.info(f"Salvaged {len(salvaged_records)} enhanced payroll records for {employee}")
            return salvaged_records
        else:
            logger.warning(f"No enhanced payroll records could be salvaged for {employee}")
            return []

    except Exception as e:
        logger.error(f"Error salvaging enhanced payroll records for {employee}: {str(e)}")
        return []


def process_enhanced_payroll(file_path: str, api_key: str) -> Dict[str, Any]:
    """Main function to process payroll file with enhanced fields"""
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

        logger.info(f"Processing enhanced payroll file: {file_path}")

        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)

        # Find all employees
        employees = find_employees_enhanced(file_content, api_key)
        if not employees:
            raise PayrollProcessingError("No employees found in the payroll data")

        # Extract enhanced payroll records for each employee
        all_records = []
        failed_employees = []

        for i, employee in enumerate(employees):
            try:
                employee_records = extract_enhanced_payroll_records(
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
            "file_processed": file_path,
            "extraction_type": "enhanced_payroll"
        }

        logger.info(f"Enhanced payroll processing complete: {len(all_records)} records from {len(employees)} employees")
        return results

    except Exception as e:
        logger.error(f"Error processing enhanced payroll: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "processing_timestamp": datetime.now().isoformat(),
            "extraction_type": "enhanced_payroll"
        }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        logger.info(f"Enhanced payroll results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def main():
    """Simple command-line usage for enhanced payroll processing"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_payroll_processor.py <input_file> [output_file]")
        print("Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "enhanced_payroll_results.json"

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Process enhanced payroll
    print(f"Processing enhanced payroll data from {input_file}...")
    results = process_enhanced_payroll(input_file, api_key)

    # Save results
    save_results(results, output_file)

    # Print summary
    if results["success"]:
        print(f"✅ Success! Processed {results['processed_employees']}/{results['total_employees']} employees")
        print(f"📊 Extracted {results['total_records']} enhanced payroll records")
        print(f"💾 Results saved to: {output_file}")

        # Show sample of fields extracted
        if results["records"]:
            sample_record = results["records"][0]
            print(f"📋 Sample fields extracted: {list(sample_record.keys())}")
    else:
        print(f"❌ Failed: {results['error']}")


if __name__ == "__main__":
    main()