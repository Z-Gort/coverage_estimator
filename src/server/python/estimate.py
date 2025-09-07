import csv
import os
import sys
import json
from pathlib import Path
import base64
from typing import Any, Dict
import anthropic
from mistralai import Mistral
from utils import analyze_claim_backup
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def analyze_individual_document_for_charges(
    document_content: Dict[str, Any],
) -> Dict[str, Any]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Analyze this document to determine if it contains itemized move-out charges, security deposit charges, or outstanding charges from a rental property.

TITLE:
{document_content.get('title', '')}

DOCUMENT CONTENT:
{document_content.get('text', '')}

Please determine:
1. Does this document enumerate/list specific charges related to move-out, security deposit deductions, outstanding tenant chargesm or very similar?
2. If yes, extract each itemized charge with its cost and description.

Look for things like:
- Security deposit itemization/disposition
- Move-out charges
- Outstanding balance itemization  
- Ledger entries showing charges
- Repair/cleaning costs
- Damage charges
- Fee breakdowns

RULES:
--ONLY include charges which are outstanding at move-out/still due. NOT charges that were already paid.
--If document shows both paid and unpaid items, look for "balance due" and include ONLY items contributing to that balance.

The document should have specific line items with costs, not just summary amounts. Note that ledgers of charges/costs aren't neccessarily showing oustanding costs."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "analyze_document_charges",
                    "description": "Submit analysis of whether document contains itemized charges and extract them",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "has_itemized_charges": {
                                "type": "boolean",
                                "description": "True if the document contains a list/enumeration of specific charges with costs",
                            },
                            "charge_items": {
                                "type": "array",
                                "description": "List of itemized charges found in the document",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "cost": {
                                            "type": "integer",
                                            "description": "Cost of the charge in dollars (whole numbers only, no cents)",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Description of the charge/item",
                                        },
                                    },
                                    "required": ["cost", "description"],
                                },
                            },
                        },
                        "required": ["has_itemized_charges", "charge_items"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "analyze_document_charges"},
        )

        # Extract the structured output
        tool_use = response.content[0]
        if tool_use.type == "tool_use" and tool_use.name == "analyze_document_charges":
            return tool_use.input  # type: ignore
        else:
            return {
                "has_itemized_charges": False,
                "charge_items": [],
                "error": "Unexpected response format",
            }

    except Exception as e:
        return {
            "has_itemized_charges": False,
            "charge_items": [],
            "error": f"API call failed: {str(e)}",
        }


def analyze_itemized_charge_coverage(charge_items, folder_contents, claim_data):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    evidence_text = ""
    for doc in folder_contents:
        evidence_text += f"\n--- {doc['title']} ---\n{doc['text']}"

    charges_text = ""
    for i, item in enumerate(charge_items, 1):
        charges_text += f"{i}. {item['description']}: ${item['cost']}\n"

    prompt = f"""Analyze these itemized charges for insurance coverage eligibility.

ITEMIZED CHARGES TO ANALYZE:
{charges_text}

SUPPORTING DOCUMENTS:
{evidence_text}

For each charge in order, determine if it's covered by insurance by analyzing the documents and following the given decision rules.

RULES:
--Any document explicitly stating claim rules OVERRIDE these general guidelines.
--Repairs, maintenance, cleaning/carpet cleaning,loss of rent due to inhability, reletting fees, are generally covered.
--Admin/other fees (EXCEPT for reletting fees), utilities, pet incurred damages, pest control, gutter cleaning, are generlly not covered.
--When in doubt, allow the charge to be covered.

"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "submit_coverage_analysis",
                    "description": "Submit coverage decision for each charge",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "coverage_decisions": {
                                "type": "array",
                                "description": f"Exactly {len(charge_items)} coverage decisions, one for each charge in order",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "covered": {
                                            "type": "boolean",
                                            "description": "True if covered by insurance, false if tenant responsibility",
                                        },
                                        "reasoning": {
                                            "type": "string",
                                            "description": "Explanation for this coverage decision",
                                        },
                                    },
                                    "required": ["covered", "reasoning"],
                                },
                                "minItems": len(charge_items),
                                "maxItems": len(charge_items),
                            }
                        },
                        "required": ["coverage_decisions"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "submit_coverage_analysis"},
        )

        tool_use = response.content[0]
        if tool_use.type == "tool_use" and tool_use.name == "submit_coverage_analysis":
            result = tool_use.input
            coverage_decisions = result.get("coverage_decisions")  # type: ignore

            # Sum covered charges
            total_covered = sum(
                charge_items[i]["cost"]
                for i, decision in enumerate(coverage_decisions)
                if decision.get("covered")
            )

            # Apply max benefit cap
            max_benefit_str = (
                claim_data.get("Max Benefit").replace("$", "").replace(",", "")
            )
            max_benefit = int(float(max_benefit_str))
            approved_benefit = min(total_covered, max_benefit)

            return {
                "approved_benefit": approved_benefit,
                "coverage_decisions": coverage_decisions,
            }
        else:
            return {"error": "Unexpected response format"}
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}


def process_claim_by_folder_number(folder_number):
    claims_dict = read_security_deposit_claims()
    claim_data = claims_dict.get(str(folder_number))

    folder_info = read_folder_contents(str(folder_number))
    folder_contents = []
    charge_items = []
    found_itemized_doc = False

    for file_info in folder_info:
        content = extract_document_content(file_info["path"])
        if "error" not in content:
            print(f"Content: {content["text"][:20]}")
            folder_contents.append(content)

            if not found_itemized_doc:
                charge_analysis = analyze_individual_document_for_charges(content)
                print(f"Charge analysis: {charge_analysis}")
                if charge_analysis.get("has_itemized_charges"):
                    charge_items = charge_analysis.get("charge_items", [])
                    found_itemized_doc = True
        else:
            print(f"Error extracting {file_info['path']}: {content['error']}")
            continue

    if found_itemized_doc:
        total_charges = sum(item["cost"] for item in charge_items)
        claim_amount_str = (
            claim_data.get("Amount of Claim").replace("$", "").replace(",", "")  # type: ignore
        )
        print(
            f"Found itemized doc, total charges: {total_charges}, claim amount: {claim_amount_str}"
        )
        try:
            claim_amount = int(float(claim_amount_str))
            if (
                total_charges >= 0.8 * claim_amount
                and total_charges <= claim_amount * 1.2
            ):  # There's one type of ledger which AI can't reliably parse
                return analyze_itemized_charge_coverage(
                    charge_items, folder_contents, claim_data
                )
            else:
                print(
                    f"Total charges {total_charges} are not close to {claim_amount}. Moving to backup."
                )
                pass
        except Exception as e:
            print(f"Error analyzing itemized charges: {e}")
            pass

    return analyze_claim_backup(folder_contents, claim_data, ANTHROPIC_API_KEY)


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


def run_unit_tests():
    folder_numbers = [365, 373, 413, 449, 455, 456]
    claims_dict = read_security_deposit_claims()

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_folder = {
            executor.submit(process_claim_by_folder_number, folder): folder
            for folder in folder_numbers
        }

        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                result = future.result()
                computed_benefit = result.get("approved_benefit", "N/A")

                # Get reference values from CSV
                claim_data = claims_dict.get(str(folder), {})
                reference_benefit = claim_data.get("Approved Benefit Amount", "N/A")
                pm_explanation = claim_data.get("PM Explanation", "N/A")

                print(
                    f"Folder {folder}: Computed=${computed_benefit} | Human=${reference_benefit} | Explan: {pm_explanation}"
                )
            except Exception as e:
                print(f"Folder {folder}: Error - {e}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "unit":
            print("Running unit tests...")
            run_unit_tests()
        else:
            folder_number = int(sys.argv[1])

            print("starting python script")
            result = process_claim_by_folder_number(folder_number)
            print(result)

            approved_benefit = result.get("approved_benefit")  # type: ignore
            # if not testing, update db
            if len(sys.argv) == 3:
                row_id = int(sys.argv[2])
                if approved_benefit is not None:
                    update_database_result(row_id, approved_benefit)

    except Exception as e:
        import traceback

        # Get detailed error information
        error_details = {
            "error": f"Script execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "line_number": (
                traceback.extract_tb(e.__traceback__)[-1].lineno
                if e.__traceback__
                else None
            ),
            "filename": (
                traceback.extract_tb(e.__traceback__)[-1].filename
                if e.__traceback__
                else None
            ),
        }

        print(json.dumps(error_details, indent=2))
        sys.exit(1)

# 365 -- partial payout, no coverage income HOA violations, covering property damages only (can't read itemized doc--will payout full)
# 373 -- normal full payout (good)
# 413 -- partial payout excluding tenant fees (good)
# 418 -- partial--excludes fees (good)
# 456 -- payout limited by max benefit (good)
# 455 -- No coverage pest/gutter (difficult) -- (good)
# 449 -- No coverage for asset protection fee or utility expenses -- (good)
# 726 -- gives full payout (can't read itemized doc--will payout full)
# 757 -- normal full payout (good)
