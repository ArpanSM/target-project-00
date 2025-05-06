rules = [
    'buy two, get a $10 Target GiftCard.',
    'For best results, follow the care instructions included in the package insert.',
    'Includes a 1‑year limited warranty and free returns within 90 days at Target.',
    'Join Target Circle to earn 1% rewards on every purchase.',
    'Now eligible for Same Day Delivery and Drive Up service in most stores.',
    'Compatible with most leading accessories (sold separately).',
    'Customers love the easy‑to‑clean surfaces and durable construction.',
    'Designed with busy families in mind, this product meets rigorous safety standards.',
    'Meets or exceeds ASTM and JPMA safety certifications.',
    'Crafted from BPA‑free materials and third‑party tested for quality assurance.'
]

import re
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from utils import call_litellm


import time
import litellm
from litellm import completion, completion_cost

def litellm(**kwargs):
    start_time = time.time()
    response = completion(**kwargs)
    output = response.choices[0].message.content
    cost = completion_cost(completion_response=response)
    formatted_string = f"${float(cost):.10f}"
    print(f'{kwargs.get("model", "unknown")} cost: {formatted_string}')
    print(f'{kwargs.get("model", "unknown")} execution time: {time.time() - start_time:.4f} seconds')
    if kwargs.get('response_format'):
        output = json.loads(output)
    return response, output

class RuleValidation(BaseModel):
    reasoning: str = Field(description="why the rule is relevant/irrelevant to the product")
    ruleValid: bool = Field(description="if the rule is valid for the product")
    ruleDescription: Optional[str] = Field(description="description of the rule when it is valid. Ex. product is bpafree, product has buy 2 get 10 gift card offer, same day delivery, etc.")

def process_product(product: Dict) -> Dict:
    # Clean text by removing headers
    _HEADER_REGEX = re.compile(r"\b(?:Description|Specifications?|Other\s+Details)\s*:\s*", flags=re.I)
    cleaned_text = _HEADER_REGEX.sub("", product["text"])
    
    validations = []
    llm_description = ""
    for rule in rules:
        if rule in cleaned_text:
            if rule == 'Join Target Circle to earn 1% rewards on every purchase.':
                validation = {
                    "reasoning": "Standard loyalty program benefit applicable to product purchase.",
                    "ruleValid": True,
                    "ruleDescription": "Earn 1% rewards with Target Circle."
                }
            elif rule == 'Now eligible for Same Day Delivery and Drive Up service in most stores.':
                validation = {
                    "reasoning": "Same Day Delivery and Drive Up are relevant services.",
                    "ruleValid": True,
                    "ruleDescription": "Same Day Delivery and Drive Up."
                }
            else:
                validation = validate_rule_with_llm(product, cleaned_text, rule)
            validations.append(validation)   
            if validation["ruleValid"]:
                llm_description += validation["ruleDescription"] + "-"

    return {**product, "llm_validations": validations, "llm_description": llm_description}

def validate_rule_with_llm(product: Dict, cleaned_text: str, rule: str) -> RuleValidation:
    prompt = f"""Analyze this product and rule combination:
    Product: {product['title']}
    Category: {product['metadata']['category']}
    Price: ${product['metadata']['price']}
    Description: {cleaned_text}
    
    Rule to validate: {rule}

    The rule could be present in the description or not.
    If the rule is present, check if it is relevant and logical for this product. Understand that this is dirty data and some rules are not relevant to the product. Think carefully.
    If the rule is not present, return ruleValid as false.
    Reasoning should be within 15 words.
    Rule Description when valid should be short and concise within 10 words.
    
    Is this rule relevant and logical for this product? Consider:
    1. Price vs offer value (e.g., $10 gift card on $15 product)
    2. Product category relevance (e.g., warranties on consumables)
    3. Logical compatibility claims
    4. Safety certifications relevance
    """
    # print(prompt)
    
    _, result = litellm(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format=RuleValidation
    )
    print(json.dumps(result, indent=4))
    print(f'==============================================')
    return result

if __name__ == "__main__":
    # Read and process products
    with open("target_products.jl") as f:
        products = [json.loads(line) for line in f]
    
    processed_products = [process_product(p) for p in products]
    
    # Write results
    with open("filtered_products.jl", "w") as f:
        for product in processed_products:
            f.write(json.dumps(product) + "\n")


