import csv
import os
import sys
import json
from pathlib import Path
import base64
import anthropic
from mistralai import Mistral
from utils import filter_documents
import psycopg2

ANTHROPIC_API_KEY = "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"

MISTRAL_API_KEY = "hnEcbqbI4cumUHOY8yew25sLjLG1Yoyb"


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
            # Remove answers
            filtered_row = {
                k: v
                for k, v in row.items()
                if k not in ["Approved Benefit Amount", "PM Explanation"]
            }
            claims_dict[tracking_number] = filtered_row

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
            content =  _extract_pdf_content(file_path)
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

    return content


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


def analyze_claim_with_anthropic(folder_contents, claim_data, api_key):
    client = anthropic.Anthropic(api_key=api_key)

    # Prepare the evidence from documents
    evidence_text = ""
    for doc in folder_contents:
        evidence_text += f"\n--- {doc['title']} ---\n"
        evidence_text += doc["text"]

    # Create the analysis prompt
    prompt = f"""You are analyzing a security deposit claim. Here is the key information:

CLAIM DETAILS:

MOST IMPORTANT:
- Max Benefit: {claim_data.get('Max Benefit', 'Not specified')}
- Amount of Claim: {claim_data.get('Amount of Claim', 'Not specified')}
OTHER INFORMATION:
- Monthly Rent: {claim_data.get('Monthly Rent', 'Not specified')}
- Lease Address: {claim_data.get('Lease Street Address', 'Not specified')}
- Lease Dates: {claim_data.get('Lease Start Date', 'Not specified')} to {claim_data.get('Lease End Date', 'Not specified')}
- Move-Out Date: {claim_data.get('Move-Out Date', 'Not specified')}
- Termination Type: {claim_data.get('Termination Type', 'Not specified')}

EVIDENCE FROM DOCUMENTS:
{evidence_text}

Please analyze this claim and determine the approved benefit amount along with your reasoning based on the following rules.

RULES:
-- The approved benefit will be between 0 and the minimum of the max benefit and the amount of claim.
-- The tenant is likely paying the security deposit monthly--that is ok! Do not consider this in your decision.
-- When in doubt, tend to trust the claim and lean toward approving reasonable amounts
-- IMPORTANT: A key element in determining what will be covered or not is understanding what items are covered by the insurer and which are solely the responsibility of the tenant.
e.g. Tenant fees, damages past the end of coverage date, etc... are not covered by the insurer. Rule of thumb is if lease paragraph allowing the charge --> tenenat responsibility.
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "submit_claim_analysis",
                    "description": "Submit the final claim analysis with approved benefit amount and reasoning",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "approved_benefit": {
                                "type": "integer",
                                "description": "The approved benefit amount in dollars (no cents, whole number only)",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Detailed explanation of the decision including analysis of evidence and factors considered",
                            },
                        },
                        "required": ["approved_benefit", "reasoning"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "submit_claim_analysis"},
        )

        # Extract the structured output
        tool_use = response.content[0]
        if tool_use.type == "tool_use" and tool_use.name == "submit_claim_analysis":
            return tool_use.input
        else:
            return {"error": "Unexpected response format from Anthropic API"}

    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}


def process_claim_by_folder_number(folder_number, api_key):
    claims_dict = read_security_deposit_claims()
    claim_data = claims_dict.get(str(folder_number))

    # Read folder contents
    folder_info = read_folder_contents(str(folder_number))
    folder_contents = []

    for file_info in folder_info:
        content = extract_document_content(file_info["path"])
        if "error" not in content:
            folder_contents.append(content)
        else:
            print(f"Warning: Could not process {file_info['name']}: {content['error']}")

    # folder_contents = filter_documents(folder_contents, api_key)

    return analyze_claim_with_anthropic(folder_contents, claim_data, api_key)


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


if __name__ == "__main__":
    try:
        folder_number = int(sys.argv[1])

        print("starting python script")
        result = process_claim_by_folder_number(folder_number, ANTHROPIC_API_KEY)
        print(result)
        
        approved_benefit = result.get("approved_benefit")
        #if not testing, update db
        if len(sys.argv) == 3:
            row_id = int(sys.argv[2])
            if approved_benefit is not None:
                update_database_result(row_id, approved_benefit)

    except Exception as e:
        print(json.dumps({"error": f"Script execution failed: {str(e)}"}))
        sys.exit(1)
