from pydantic import BaseModel
from typing import Optional


# Pydantic model for Mistral OCR document annotation
class ChargeItem(BaseModel):
    cost: float  # Cost of the charge in dollars
    description: str  # Description of the charge/item
    date: Optional[str] = (
        None  # Optional date associated with the charge (mm/dd/yy format)
    )
    is_rent: bool = False  # Represents unpaid rent


EXCALIBUR_DOCSTRING = """
    You are given a ledger showing both paid and oustanding charges of the tenant. You should analyze this document and extract out all the unpaid charges.

    FORMAT OF THE DOCUMENT:
    --There will be headers with summary information--note "balance due"--our oustanding charges should approximately add to this
    --There will be many charges listed in chronological order going downward--THE OUTSTANDING CHARGES ARE GENERALLY AT THE BOTTOM
    
    HOW TO FIND THE OUTSTANDING CHARGES:
    --Find the LOWEST negative charge, get ALL the charges below that
      --(Exception) if the negative charge is the final charge, use the second to last negative charge

    THE KEY POINT:
    --DONT add all the charges--only those BELOW the last negative charge

    Charges will be things like:
    - Security deposit itemization/disposition
    - Move-out charges
    - Outstanding balance itemization
    - Ledger entries showing charges
    - Repair/cleaning costs
    - Damage charges
    - Fee breakdowns
    - Pending unpaid rent
    """

PURE_OPERATING_LLC_DOCSTRING = """
    You are given a ledger showing both paid and oustanding charges of the tenant. You should analyze this document and extract out all the unpaid charges.

    FORMAT OF THE DOCUMENT:
    --There will be headers with summary information--note "Total Unpaid"--the sum of the unpaid charges you find should approximately sum to this
    --There will be charges against the tenant and payments from the tenant listed in the table where newer payments are at the top
    
    HOW TO FIND THE OUTSTANDING CHARGES:
    --Find the NEWEST (highest) payment from the tenant
      --IMPORTANT PAYMENT INDICATOR: The number in the balance column will go DOWN to the one below it
      --Side indicator: look for a number in the payments field (this is a sign, but not always accurate)
    --Anything listed as an unpaid charge OR returned payment adjustment above the newest payment IS AN OUSTANDING CHARGE
      --Note: returned payment adjustments SHOULD be considered an outstanding charge IF they are above the newest payment

    THE KEY POINT:
    --DONT add all charges or returned payment adjustments as ChargeItems--ONLY those above the newest payment

    """
  
DEFAULT_DOCSTRING = """
  Analyze this document to determine if it contains itemized move-out charges, security deposit charges, or outstanding charges from a rental property. ONLY UNPAID CHARGES SHOULD BE ADDED.

  Look for things like:
  - Security deposit itemization/disposition
  - Move-out charges
  - Outstanding balance itemization
  - Ledger entries showing charges
  - Repair/cleaning costs
  - Damage charges
  - Fee breakdowns
  - Pending unpaid rent

  RULES:
  --ONLY include charges which are outstanding at move-out/still due. NOT charges that were already paid.
  --If document shows both paid and unpaid items, look items which still have UNPAID balance.

  NOTE:
  --The document should have specific line items with costs, not just summary amounts. Note that ledgers of charges/costs aren't necessarily showing outstanding costs.
  --Pay attention to decimals and commas. (Numbers will barely ever be above 10,000 in reality)

  DATE EXTRACTION:
  --For each charge, try to find an associated date if possible and write in mm/dd/yy format (If not found leave None).

  IS_RENT:
  --ONLY if you are confident a charge represents unpaid rent, set is_rent to True.
  """


def create_analysis_class(mgmt_company: str = ""):
    docstring = DEFAULT_DOCSTRING
    if mgmt_company == "Excalibur Homes":
        docstring = EXCALIBUR_DOCSTRING
    elif mgmt_company == "Pure Operating LLC":
        docstring = PURE_OPERATING_LLC_DOCSTRING

    class CustomChargeAnalysis(BaseModel):
        has_itemized_charges: bool
        charge_items: list[ChargeItem]

    # Set the custom docstring
    CustomChargeAnalysis.__doc__ = docstring

    return CustomChargeAnalysis
