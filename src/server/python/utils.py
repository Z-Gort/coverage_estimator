import csv
import os
from pathlib import Path
import base64
import anthropic
import json
from mistralai import Mistral


# backup
def analyze_claim_backup(folder_contents, claim_data, api_key):
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
