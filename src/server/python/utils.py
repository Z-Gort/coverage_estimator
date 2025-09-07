import csv
from pathlib import Path
import base64
import anthropic
from mistralai import Mistral
import psycopg2
import math

# API Keys
ANTHROPIC_API_KEY = "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"
MISTRAL_API_KEY = "hnEcbqbI4cumUHOY8yew25sLjLG1Yoyb"


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


def filter_claim_data_for_ai(claim_data):
    """Remove reference answers from claim data before sending to AI"""
    return {
        k: v
        for k, v in claim_data.items()
        if k not in ["Approved Benefit Amount", "PM Explanation"]
    }


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
    covered_amount: int, max_benefit: int, monthly_rent: int | None = None
) -> int:
    """Calculate approved benefit considering max benefit and optional monthly rent ceiling"""
    print(f"Calculating approved benefit for covered amount: {covered_amount}, max benefit: {max_benefit}, monthly rent: {monthly_rent}")
    if monthly_rent:
        monthly_rent_ceiling = calculate_monthly_rent_ceiling(monthly_rent)
        return min(covered_amount, max_benefit, monthly_rent_ceiling)
    else:
        return min(covered_amount, max_benefit)