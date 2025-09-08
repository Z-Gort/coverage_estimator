import sys
import json
import time
import os
import tempfile
from typing import Any, Dict
from pathlib import Path
import anthropic
from pydantic import BaseModel
from mistralai import Mistral, DocumentURLChunk
from mistralai.extra import response_format_from_pydantic_model
from utils import (
    read_security_deposit_claims,
    read_folder_contents,
    update_database_result,
    ANTHROPIC_API_KEY,
    MISTRAL_API_KEY,
    calculate_approved_benefit,
    encode_pdf_to_base64,
    count_pdf_pages,
    clip_pdf_to_pages,
)


# Pydantic model for Mistral OCR document annotation
class ChargeItem(BaseModel):
    cost: int  # Cost of the charge in dollars (whole numbers only, no cents)
    description: str  # Description of the charge/item


class DocumentChargeAnalysis(BaseModel):
    """
    Analyze this document to determine if it contains itemized move-out charges, security deposit charges, or outstanding charges from a rental property.

    Look for things like:
    - Security deposit itemization/disposition
    - Move-out charges
    - Outstanding balance itemization
    - Ledger entries showing charges
    - Repair/cleaning costs
    - Damage charges
    - Fee breakdowns
    - Monthly rent amounts (from lease agreements, rent schedules, ledgers showing monthly rent payments)

    RULES:
    --ONLY include charges which are outstanding at move-out/still due. NOT charges that were already paid.
    --If document shows both paid and unpaid items, look for "balance due" and include ONLY items contributing to that balance.

    The document should have specific line items with costs, not just summary amounts. Note that ledgers of charges/costs aren't necessarily showing outstanding costs.

    SPECIAL CASE:
    If the document is called ledger and has a 'balance as of' and 'total unpaid' in the grid. Then look to grab all the charges AFTER the last paid rent.
    """

    has_itemized_charges: bool  # True if the document contains a clear list/enumeration of specific charges with individual costs
    charge_items: list[ChargeItem]  # Array of itemized charges found in the document
    monthly_rent: (
        int | None
    )  # Monthly rent amount in dollars if found in the document, null if not found


def analyze_individual_document_for_charges_ocr(file_path: str) -> Dict[str, Any]:
    """Analyze document using Mistral OCR with document annotations."""
    try:
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()

        # Clip PDF if >8 pages
        pdf_to_process = file_path
        if file_extension == ".pdf" and count_pdf_pages(file_path) > 8:
            print(f"Clipping PDF to 8 pages")
            temp_fd, pdf_to_process = tempfile.mkstemp(suffix=".pdf", prefix="clipped_")
            os.close(temp_fd)
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
        else:
            return {
                "has_itemized_charges": False,
                "charge_items": [],
                "monthly_rent": None,
                "error": f"Unsupported file type",
            }

        # Process with Mistral OCR
        client = Mistral(api_key=MISTRAL_API_KEY)
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document=DocumentURLChunk(document_url=document_url),
            document_annotation_format=response_format_from_pydantic_model(
                DocumentChargeAnalysis
            ),
            include_image_base64=True,
        )

        # Extract annotation data
        annotation_data = getattr(response, "document_annotation", None)
        if isinstance(annotation_data, str):
            annotation_data = json.loads(annotation_data)

        if annotation_data:
            charge_items = [
                {"cost": int(item["cost"]), "description": str(item["description"])}
                for item in annotation_data.get("charge_items", [])
            ]
            monthly_rent = annotation_data.get("monthly_rent")
            return {
                "has_itemized_charges": bool(
                    annotation_data.get("has_itemized_charges", False)
                ),
                "charge_items": charge_items,
                "monthly_rent": int(monthly_rent) if monthly_rent else None,
            }
        else:
            return {
                "has_itemized_charges": False,
                "charge_items": [],
                "monthly_rent": None,
                "error": "No annotation data found",
            }

    except Exception as e:
        return {
            "has_itemized_charges": False,
            "charge_items": [],
            "monthly_rent": None,
            "error": f"OCR failed: {str(e)}",
        }


def analyze_itemized_charge_coverage(charge_items, claim_data, monthly_rent=None):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    charges_text = ""
    for i, item in enumerate(charge_items, 1):
        charges_text += f"{i}. {item['description']}: ${item['cost']}\n"

    prompt = f"""Analyze these itemized charges for insurance coverage eligibility.

ITEMIZED CHARGES TO ANALYZE:
{charges_text}

For each charge in order, determine if it's covered by insurance following the given decision rules.

RULES:
--Repairs, maintenance, cleaning/carpet cleaning, loss of rent due to inability, reletting fees, unpaid rent, are generally covered.
--Admin/other fees, utilities, pet incurred damages, pest control, gutter cleaning, are generally NOT covered (anything not listed is generally covered).
--When in doubt, COVER THE CHARGE.

"""
    # Note: Unpaid rent seems to be covered and not covered in different cases

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

            # Get claim amount
            claim_amount_str = (
                claim_data.get("Amount of Claim").replace("$", "").replace(",", "")
            )
            claim_amount = int(float(claim_amount_str))

            approved_benefit = calculate_approved_benefit(
                total_covered, max_benefit, claim_amount, monthly_rent
            )

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
    charge_items = []
    found_itemized_doc = False
    monthly_rent = None
    found_monthly_rent = False

    for file_info in folder_info:
        # Analyze document with OCR for both charges and monthly rent
        charge_analysis = analyze_individual_document_for_charges_ocr(file_info["path"])
        print(f"Charge analysis: {charge_analysis}")

        # Check for itemized charges if not found yet
        if not found_itemized_doc and charge_analysis.get("has_itemized_charges"):
            charge_items = charge_analysis.get("charge_items", [])
            found_itemized_doc = True

        # Check for monthly rent if not found yet
        if not found_monthly_rent and charge_analysis.get("monthly_rent"):
            monthly_rent = charge_analysis.get("monthly_rent")
            found_monthly_rent = True

        # Stop processing if we found both
        if found_itemized_doc and found_monthly_rent:
            break

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
            ):  # Note: there are one or two docs the AI can't reliably parse--so total_charges can be off.
                return analyze_itemized_charge_coverage(
                    charge_items, claim_data, monthly_rent
                )
            else:
                print(
                    f"Total charges {total_charges} are not close to {claim_amount}. Moving to backup."
                )
                pass
        except Exception as e:
            print(f"Error analyzing itemized charges: {e}")
            pass

    max_benefit_str = (
        claim_data.get("Max Benefit").replace("$", "").replace(",", "")  # type: ignore
    )
    max_benefit = int(float(max_benefit_str))

    requested_claim_str = (
        claim_data.get("Amount of Claim").replace("$", "").replace(",", "")  # type: ignore
    )
    requested_claim = int(float(requested_claim_str))

    return {
        "approved_benefit": calculate_approved_benefit(
            max_benefit, max_benefit, requested_claim, monthly_rent
        )
    }


def process_claims_batch(folder_numbers, row_id=None):
    result_list = []
    claims_dict = read_security_deposit_claims()

    for i, folder_number in enumerate(folder_numbers):
        try:
            claim_data = claims_dict.get(str(folder_number))
            if (
                not claim_data
                or not claim_data["Approved Benefit Amount"]
                or not claim_data["Amount of Claim"]
            ):
                print("Invalid row: ", folder_number)
                continue

            folder_path = Path(str(folder_number))
            if not folder_path.exists() or not folder_path.is_dir():
                print(f"Folder {folder_number} does not exist, skipping...")
                continue

            result = process_claim_by_folder_number(folder_number)
            print(
                "Result for folder ", folder_number, ": ", json.dumps(result, indent=2)
            )
            ai_approved_benefit = result.get("approved_benefit") if result else 0
            actual_approved_benefit_str = (
                claim_data.get("Approved Benefit Amount") if claim_data else "$0"
            )
            actual_approved_benefit = int(
                float(actual_approved_benefit_str.replace("$", "").replace(",", ""))
            )
            pm_explanation = claim_data.get("PM Explanation") if claim_data else None

            print(
                f"Folder {folder_number}: AI=${ai_approved_benefit}, Actual=${actual_approved_benefit}, PM={pm_explanation}"
            )

            result_list.extend(
                [folder_number, ai_approved_benefit, actual_approved_benefit]
            )

        except Exception as e:
            print(f"Error processing folder {folder_number}: {str(e)}")

    if row_id:
        update_database_result(row_id, result_list)


if __name__ == "__main__":
    try:
        print("starting python script")
        if len(sys.argv) == 3:
            # Called from estimate.ts: row_id, folder_numbers_str
            row_id = int(sys.argv[1])
            folder_numbers_str = sys.argv[2]
            folder_numbers = [int(num.strip()) for num in folder_numbers_str.split(",")]
            process_claims_batch(folder_numbers, row_id)
        else:
            # Local testing: just folder_numbers_str
            folder_numbers_str = sys.argv[1]
            folder_numbers = [int(num.strip()) for num in folder_numbers_str.split(",")]
            process_claims_batch(folder_numbers)

    except Exception as e:
        print(f"Script failed: {str(e)}")
        sys.exit(1)


# CLAUDE:
# 365 -- partial payout, no coverage income HOA violations, covering property damages only (can't read itemized doc--will payout ful OR payout tiny--BAD ERROR)
# 373 -- normal full payout -- fails to read, tries to round to rent (error of 20%--not terrible)
# 405 -- AI thinks it shouldn't pay out for unpaid rent (big error I think)
# 413 -- partial payout excluding tenant fees (good)
# 417 -- partial--excludes fees (good)
# 449 -- No coverage for asset protection fee or utility expenses -- (quite close)
# 455 -- No coverage pest/gutter -- (quite close)
# 456 -- payout limited by max benefit (good)
# 703 -- constrained by monthly rent (good)
# 705 -- constrained by montly rent (good)
# 726 -- gives full payout (can't read itemized, pays out full--turns out to be low error))
# 728 -- partial excluding fees (correctly excludes fees--good)
# 757 -- normal full payout (pays out full--good)
