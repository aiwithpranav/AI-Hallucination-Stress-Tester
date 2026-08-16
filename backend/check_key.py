import os
from dotenv import load_dotenv

load_dotenv('../.env', override=True)
key = os.getenv('GEMINI_API_KEY', '')
length = len(key)
has_spaces = ' ' in key
is_placeholder = key == 'your_gemini_api_key_here'
print(f"Key loaded. Length: {length}, Has spaces: {has_spaces}, Is placeholder: {is_placeholder}")
