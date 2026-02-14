import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_delays(email_text):
    """
    Extracts trailer delay information from email text using Gemini API with a regex fallback.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize default result
    result = {
        "trailer_id": None,
        "parts_days": 0,
        "accident_days": 0,
        "training_days": 0
    }

    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('models/gemini-2.5-flash')
            
            prompt = f"""

            You are extracting manufacturing delays and other specifications info from an email.

            Extract trailer repair delay information from the following email text.
            Return ONLY a valid JSON object with the following keys:
            "trailer_id" (string, e.g., "T103"),
            "parts_days" (integer),
            "accident_days" (integer),
            "training_days" (integer).
            
            Rules:
            - trailer_id looks like T### (e.g., T003).
            - If the email says "delayed by X days because of missing parts" => parts_days = X.
            - If it says "missing parts" without days, infer parts_days = 1.
            - If it mentions "accident" with days, set accident_days accordingly; if accident mentioned with no days, accident_days = 1.
            - If it mentions "training" with days, set training_days accordingly; if training mentioned with no days, training_days = 1.
            - If it says "delayed by X days" and category is unclear, set parts_days = X (default bucket).
            - If a category is not mentioned, use 0.

            If a field is missing, return 0. "trailer_id" is required.
            
            Email Text:
            {email_text}
            """
            
            response = model.generate_content(prompt)

            ##adding check
            print("RAW GEMINI RESPONSE >>>", response.text)
            ##adding check

            # Extract JSON from response text (handling potential markdown formatting)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                result.update(extracted_data)
                return result
        except Exception as e:
            print(f"Gemini API Error: {e}")
    
    # Regex Fallback if Gemini fails or API key is missing
    # Look for Trailer ID like T followed by digits
    id_match = re.search(r'T\d+', email_text, re.IGNORECASE)
    if id_match:
        result["trailer_id"] = id_match.group().upper()
    
    # Simple regex for days (e.g., "3 days for parts", "accident: 2 days")
    parts_match = re.search(r'(\d+)\s*days?\s*for\s*parts', email_text, re.IGNORECASE)
    if parts_match:
        result["parts_days"] = int(parts_match.group(1))
        
    accident_match = re.search(r'accident[:\s]+(\d+)\s*days?', email_text, re.IGNORECASE)
    if accident_match:
        result["accident_days"] = int(accident_match.group(1))
        
    training_match = re.search(r'training[:\s]+(\d+)\s*days?', email_text, re.IGNORECASE)
    if training_match:
        result["training_days"] = int(training_match.group(1))

    return result
