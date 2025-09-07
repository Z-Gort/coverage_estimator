import sys
import json
import math
from typing import Any, Dict
import anthropic
from utils import (
    analyze_claim_backup,
    read_security_deposit_claims,
    read_folder_contents,
    extract_document_content,
    update_database_result,
    ANTHROPIC_API_KEY,
    MISTRAL_API_KEY,
)
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def analyze_document_for_monthly_rent(document_content: Dict[str, Any]) -> int | None:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""Look for any monthly rent amount in this document.

TITLE: {document_content.get('title', '')}
DOCUMENT CONTENT: {document_content.get('text', '')}

Find any monthly rent amount mentioned (from lease agreements, rent schedules, ledgers showing monthly rent payments, etc.)."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "extract_monthly_rent",
                    "description": "Extract monthly rent amount if found",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "monthly_rent": {
                                "type": "integer",
                                "description": "Monthly rent amount in dollars, null if not found",
                            }
                        },
                        "required": ["monthly_rent"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "extract_monthly_rent"},
        )

        tool_use = response.content[0]
        if tool_use.type == "tool_use" and tool_use.name == "extract_monthly_rent":
            result = tool_use.input
            result.get("monthly_rent")  # type: ignore
    except Exception:
        return None


def analyze_itemized_charge_coverage(
    charge_items, folder_contents, claim_data, monthly_rent=None
):
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

            # Apply monthly rent ceiling if available (rounded up to nearest $500)
            if monthly_rent:
                monthly_rent_ceiling = math.ceil(monthly_rent / 500) * 500
                approved_benefit = min(total_covered, max_benefit, monthly_rent_ceiling)
            else:
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
    monthly_rent = None
    found_monthly_rent = False

    for file_info in folder_info:
        content = extract_document_content(file_info["path"])
        if "error" not in content:
            print(f"Content: {content["text"][:20]}")
            folder_contents.append(content)

            # Look for itemized charges if not found yet
            if not found_itemized_doc:
                charge_analysis = analyze_individual_document_for_charges(content)
                print(f"Charge analysis: {charge_analysis}")
                if charge_analysis.get("has_itemized_charges"):
                    charge_items = charge_analysis.get("charge_items", [])
                    found_itemized_doc = True

            # Look for monthly rent if not found yet
            if not found_monthly_rent:
                doc_monthly_rent = analyze_document_for_monthly_rent(content)
                if doc_monthly_rent:
                    monthly_rent = doc_monthly_rent
                    found_monthly_rent = True

            # Stop processing if we found both
            if found_itemized_doc and found_monthly_rent:
                break
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
            ):  # Note: there are one or two docs the AI can't reliably parse--so total_charges can be off
                return analyze_itemized_charge_coverage(
                    charge_items, folder_contents, claim_data, monthly_rent
                )
            else:
                print(
                    f"Total charges {total_charges} are not close to {claim_amount}. Moving to backup."
                )
                pass
        except Exception as e:
            print(f"Error analyzing itemized charges: {e}")
            pass

    return analyze_claim_backup(folder_contents, claim_data)


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
                computed_benefit = (
                    result.get("approved_benefit", "N/A")
                    if isinstance(result, dict)
                    else "N/A"
                )

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
# 417 -- partial--excludes fees (good)
# 456 -- payout limited by max benefit (good)
# 455 -- No coverage pest/gutter (difficult) -- (good)
# 449 -- No coverage for asset protection fee or utility expenses -- (good)
# 726 -- gives full payout (can't read itemized doc--will payout full)
# 727 -- partial excluding fees (good)
# 757 -- normal full payout (good)
# 705 -- cap on monthly rent (good)
