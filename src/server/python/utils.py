import csv
from pathlib import Path
import base64
from mistralai import Mistral
import psycopg2
import math
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
from typing import Optional


# API Keys
ANTHROPIC_API_KEY = "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"
MISTRAL_API_KEY = "hnEcbqbI4cumUHOY8yew25sLjLG1Yoyb"


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


def encode_image_to_base64(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error: {e}")
        return None


def extract_document_content(file_path):
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    try:
        if file_extension == ".pdf":
            content = _extract_pdf_content(file_path)
            return content
        elif file_extension in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            content = _extract_image_content(file_path)
            return content
        else:
            return {"error": f"Unsupported file type: {file_extension}"}
    except Exception as e:
        return {"error": f"Failed to extract content: {str(e)}"}


def _extract_pdf_content(file_path):
    content = {
        "title": Path(file_path).name,
        "file_path": str(file_path),
        "file_type": "pdf",
        "text": "",
    }

    # Encode PDF to base64
    base64_pdf = encode_pdf_to_base64(file_path)

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
            include_image_base64=True,
        )

        content["text"] = getattr(ocr_response.pages[0], "markdown", "")

        return content

    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return {"error": f"Mistral OCR failed: {str(e)}"}


def _extract_image_content(file_path):
    content = {
        "title": Path(file_path).name,
        "file_path": str(file_path),
        "file_type": "image",
        "text": "",
    }

    base64_image = encode_image_to_base64(file_path)

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)

        # Determine image format for proper MIME type
        file_extension = Path(file_path).suffix.lower()
        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tiff": "image/tiff",
            ".bmp": "image/bmp",
        }
        mime_type = mime_type_map.get(file_extension, "image/jpeg")

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{base64_image}",
            },
            include_image_base64=True,
        )

        content["text"] = getattr(ocr_response.pages[0], "markdown", "")

        return content

    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return {"error": f"Mistral OCR failed: {str(e)}"}


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
