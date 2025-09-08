import sys
import json
import time
from typing import Any, Dict
import anthropic
from utils import (
    read_security_deposit_claims,
    read_folder_contents,
    extract_document_content,
    update_database_result,
    ANTHROPIC_API_KEY,
    calculate_approved_benefit,
)


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
        print(f"Monthly rent tool use: {tool_use}")
        if tool_use.type == "tool_use" and tool_use.name == "extract_monthly_rent":
            result = tool_use.input  # type: ignore
            monthly_rent = result.get("monthly_rent")  # type: ignore
            # Handle cases where AI returns string "null" instead of None
            if monthly_rent == "null" or monthly_rent is None:
                return None
            # Ensure we return an integer if it's a valid number
            try:
                return int(monthly_rent)
            except (ValueError, TypeError):
                return None
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
--Admin/other fees (EXCEPT for reletting fees), utilities, unpaid rent, pet incurred damages, pest control, gutter cleaning, are generally NOT covered.
--When in doubt, allow the charge to be covered.

"""
    # Note: Unpaid rent seems to be covered and not covered in different cases.

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

            approved_benefit = calculate_approved_benefit(
                total_covered, max_benefit, monthly_rent
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
    folder_contents = []
    charge_items = []
    found_itemized_doc = False
    monthly_rent = None
    found_monthly_rent = False

    for file_info in folder_info:
        content = extract_document_content(file_info["path"])
        print(f"File content: {content}")
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
            ):  # Note: there are one or two docs the AI can't reliably parse--so total_charges can be off.
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

    max_benefit_str = (
        claim_data.get("Max Benefit").replace("$", "").replace(",", "")  # type: ignore
    )
    max_benefit = int(float(max_benefit_str))

    return {
        "approved_benefit": calculate_approved_benefit(
            max_benefit, max_benefit, monthly_rent
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

        if i < len(folder_numbers) - 1:
            time.sleep(10)

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

# 365 -- partial payout, no coverage income HOA violations, covering property damages only (can't read itemized doc--will payout full)
# 373 -- normal full payout (good)
# 405 -- AI thinks it shouldn't pay out for unpaid rent (in another case this is correct behavior)
# 413 -- partial payout excluding tenant fees (good)
# 417 -- partial--excludes fees (good)
# 456 -- payout limited by max benefit (good)
# 455 -- No coverage pest/gutter -- (good)
# 449 -- No coverage for asset protection fee or utility expenses -- (good)
# 726 -- gives full payout (can't read itemized doc--will payout full)
# 727 -- partial excluding fees (good)
# 757 -- normal full payout (good)
# 703 -- constrained by monthly rent
# 705 -- constrained by montly rent (good)
