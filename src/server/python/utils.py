import csv
from pathlib import Path
import base64
from mistralai import Mistral
import psycopg2
import math
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
from typing import Optional, Dict, Any
from mistralai import Mistral, DocumentURLChunk
from mistralai.extra import response_format_from_pydantic_model
import tempfile
import os
from dynamic_analysis import create_analysis_class
import json

# API Keys
ANTHROPIC_API_KEY = "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"
MISTRAL_API_KEY = "hnEcbqbI4cumUHOY8yew25sLjLG1Yoyb"


def analyze_individual_document_for_charges_ocr(
    file_path: str, custom_analysis_class=None
) -> Dict[str, Any]:
    """Analyze document using Mistral OCR with document annotations."""
    temp_pdf_path = None
    try:
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()

        # Clip PDF if >8 pages
        pdf_to_process = file_path
        if file_extension == ".pdf" and count_pdf_pages(file_path) > 8:
            temp_fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", prefix="clipped_")
            os.close(temp_fd)
            pdf_to_process = temp_pdf_path
            clip_pdf_to_pages(file_path, pdf_to_process, max_pages=8)

        # Encode to base64
        if file_extension == ".pdf":
            base64_data = encode_pdf_to_base64(pdf_to_process)
            document_url = f"data:application/pdf;base64,{base64_data}"
        elif file_extension in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            with open(file_path, "rb") as f:
                import base64

                base64_data = base64.b64encode(f.read()).decode("utf-8")
            mime_type = (
                f"image/{file_extension[1:]}"
                if file_extension != ".jpg"
                else "image/jpeg"
            )
            document_url = f"data:{mime_type};base64,{base64_data}"
        elif file_extension == ".docx":
            with open(file_path, "rb") as f:
                import base64

                base64_data = base64.b64encode(f.read()).decode("utf-8")
            document_url = f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{base64_data}"
        else:
            return {
                "has_itemized_charges": False,
                "charge_items": [],
                "error": f"Unsupported file type",
            }

        # Process with Mistral OCR
        client = Mistral(api_key=MISTRAL_API_KEY)
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document=DocumentURLChunk(document_url=document_url),
            document_annotation_format=response_format_from_pydantic_model(
                create_analysis_class()
                if not custom_analysis_class
                else custom_analysis_class
            ),
            include_image_base64=True,
        )

        # Extract annotation data
        annotation_data = getattr(response, "document_annotation", None)
        if isinstance(annotation_data, str):
            annotation_data = json.loads(annotation_data)

        if annotation_data:
            charge_items = [
                {
                    "cost": int(item["cost"]),
                    "description": str(item["description"]),
                    "date": item.get("date"),  # Include optional date
                    "is_rent": item.get("is_rent", False),  # Include is_rent field
                }
                for item in annotation_data.get("charge_items", [])
            ]
            return {
                "has_itemized_charges": bool(
                    annotation_data.get("has_itemized_charges", False)
                ),
                "charge_items": charge_items,
            }
        else:
            return {
                "has_itemized_charges": False,
                "charge_items": [],
                "error": "No annotation data found",
            }

    except Exception as e:
        return {
            "has_itemized_charges": False,
            "charge_items": [],
            "error": f"OCR failed: {str(e)}",
        }
    finally:
        # Clean up temporary PDF file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)


def count_pdf_pages(file_path: str) -> int:
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            return len(pdf_reader.pages)
    except Exception as e:
        print(f"Error counting pages in {file_path}: {e}")
        return -1


def clip_pdf_to_pages(input_path: str, output_path: str, max_pages: int = 8) -> bool:
    try:
        with open(input_path, "rb") as input_file:
            pdf_reader = PdfReader(input_file)
            pdf_writer = PdfWriter()

            # Add up to max_pages pages to the writer
            pages_to_add = min(len(pdf_reader.pages), max_pages)
            for i in range(pages_to_add):
                pdf_writer.add_page(pdf_reader.pages[i])

            # Write the clipped PDF
            with open(output_path, "wb") as output_file:
                pdf_writer.write(output_file)

        print(f"Successfully clipped PDF to {pages_to_add} pages: {output_path}")
        return True

    except Exception as e:
        print(f"Error clipping PDF {input_path}: {e}")
        return False


def update_database_result(row_id, result_value):
    try:
        # Get DATABASE_URL from environment or use default
        database_url = (
            "postgresql://postgres:K-aFfRISnScft1hQ@localhost:5432/corgi_fullstack"
        )

        conn = psycopg2.connect(database_url)
        cur = conn.cursor()

        cur.execute(
            "UPDATE corgi_fullstack_post SET result = %s WHERE id = %s",
            (result_value, row_id),
        )

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database update failed: {e}")


def read_folder_contents(folder_path):
    folder_path = Path(folder_path)

    files_info = []

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            file_info = {
                "name": file_path.name,
                "path": str(file_path),
                "extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
            }
            files_info.append(file_info)

    return files_info


def read_security_deposit_claims():
    claims_dict = {}

    with open(
        "Security Deposit Claims - Security Deposit Claims.csv", "r", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            tracking_number = row["Tracking Number"]
            claims_dict[tracking_number] = row

    return claims_dict


def encode_pdf_to_base64(pdf_path):
    """Encode the PDF to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_monthly_rent_ceiling(monthly_rent: int) -> int:
    """Calculate monthly rent ceiling rounded up to nearest $500"""
    return math.ceil(monthly_rent / 500) * 500


def calculate_approved_benefit(
    covered_amount: int,
    max_benefit: int,
    claim_amount: int,
    monthly_rent: int | None = None,
    claims_data: dict | None = None,
) -> int:
    """Calculate approved benefit considering max benefit, optional monthly rent ceiling, and claim amount limit"""
    constraints = [covered_amount, max_benefit, claim_amount]

    # Only apply monthly rent restriction if Treaty # is T00002 or T00001
    if monthly_rent:
        treaty_num = claims_data.get("Treaty #", "") if claims_data else ""
        if treaty_num in ["T00002", "T00001"]:
            monthly_rent_ceiling = calculate_monthly_rent_ceiling(monthly_rent)
            constraints.append(monthly_rent_ceiling)

    return min(constraints)


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse date string in various formats (mm/dd/yy, mm/dd/yyyy) to datetime object."""
    if not date_str:
        return None

    # Clean the date string
    date_str = date_str.strip()

    # Try different date formats
    formats = [
        "%m/%d/%y",  # mm/dd/yy
        "%m/%d/%Y",  # mm/dd/yyyy
        "%m-%d-%y",  # mm-dd-yy
        "%m-%d-%Y",  # mm-dd-yyyy
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def get_charge_items(
    folder_info,
    claim_amount,
    claim_data,
):
    charge_items = []
    found_itemized_doc = False
    best_diff = float("inf")

    for file_info in folder_info:
        # Analyze document with OCR for charges only
        if claim_data.get("Property Management Company", "") == "Excalibur Homes" and (
            "ledger" in file_info["path"].lower()
            or "statement" in file_info["path"].lower()
        ):
            charge_analysis = analyze_individual_document_for_charges_ocr(
                file_info["path"], create_analysis_class("Excalibur Homes")
            )
        elif (
            claim_data.get("Property Management Company", "") == "Pure Operating LLC"
            and "ledger" in file_info["path"].lower()
        ):
            charge_analysis = analyze_individual_document_for_charges_ocr(
                file_info["path"], create_analysis_class("Pure Operating LLC")
            )
        else:
            charge_analysis = analyze_individual_document_for_charges_ocr(
                file_info["path"], create_analysis_class()
            )

        # Check for itemized charges
        if charge_analysis.get("has_itemized_charges"):
            current_charge_items = charge_analysis.get("charge_items", [])
            current_total = sum(item["cost"] for item in current_charge_items)

            # Optimization--look for best itemized charges
            if claim_amount:
                current_diff = abs(current_total - claim_amount)
                if current_diff < best_diff:
                    charge_items = current_charge_items
                    found_itemized_doc = True
                    best_diff = current_diff
                    print(
                        f"New best match: total ${current_total}, diff ${current_diff}"
                    )
            elif not found_itemized_doc:
                charge_items = current_charge_items
                found_itemized_doc = True

    return charge_items, found_itemized_doc
