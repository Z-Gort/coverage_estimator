import csv
import os
from pathlib import Path
import base64
import anthropic
import json
from mistralai import Mistral


def is_security_deposit_waiver(document, api_key):
    """
    Use Claude Haiku to determine if a document contains a security deposit waiver selection.
    Returns True if the document contains security deposit payment options, False otherwise.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Create a prompt to identify security deposit waivers
    prompt = f"""You are analyzing a document to determine if it contains a "security deposit waiver" selection section.

Look for any section within the document that allows tenants to choose between:
1. Paying the security deposit in full upfront, OR  
2. Paying the security deposit in monthly installments

This selection could be:
- A standalone waiver document
- A section embedded within a lease agreement  
- Part of any rental document with checkboxes/options for security deposit payment method
- Any form where tenants can select how they want to pay their security deposit

DOCUMENT FILENAME: {document.get('title', 'Unknown')}

DOCUMENT CONTENT:
{document.get('text', '')}

Based on the filename and content, does this document CONTAIN a security deposit waiver selection (even if it's just one section of a larger document)? Look for actual selection options, checkboxes, or clear choices between payment methods.

Mark as true if there are specific options for tenants to choose their security deposit payment method, even if embedded within a larger document."""

    print("Prompt", prompt)

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-latest",  # Lightweight model
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": "classify_document",
                    "description": "Classify whether the document contains a security deposit waiver selection",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "is_security_deposit_waiver": {
                                "type": "boolean",
                                "description": "True if this document contains options for tenants to choose their security deposit payment method (full payment vs installments), False otherwise",
                            }
                        },
                        "required": ["is_security_deposit_waiver"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "classify_document"},
        )

        # Extract the boolean result from the tool use
        tool_use = response.content[0]
        print("CHECKING RESULT", tool_use)
        result = getattr(tool_use, "input", {})
        if isinstance(result, dict):
            return result.get("is_security_deposit_waiver", False)
        return False

    except Exception as e:
        print(f"Error checking document {document.get('title', 'Unknown')}: {e}")
        return False


def filter_documents(folder_contents, api_key):
    filtered_contents = []

    for document in folder_contents:
        print("Checking document: ", document.get("title", "Unknown"))
        if is_security_deposit_waiver(document, api_key):
            print(
                f"Found and filtering out security deposit waiver: {document.get('title', 'Unknown')}"
            )
            # Skip this document (don't add to filtered_contents)
            continue

        filtered_contents.append(document)

    return filtered_contents
