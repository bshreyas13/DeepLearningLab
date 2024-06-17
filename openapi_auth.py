# get open api key
import os
# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded correctly
if not openai_api_key:
    raise ValueError("API key not found. Please check your .env file.")
