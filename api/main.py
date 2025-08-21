import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from api.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required environment variables
# OPENAI_API_KEY is optional if using Azure embeddings or Ollama
google_key_missing = not os.environ.get('GOOGLE_API_KEY')
openai_key_missing = not os.environ.get('OPENAI_API_KEY')

if google_key_missing:
    logger.warning("Missing environment variable: GOOGLE_API_KEY. Google Gemini models will be unavailable.")

if openai_key_missing:
    # Only warn if neither Azure nor Ollama embedder appears configured
    azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT') or os.environ.get('API_BASE')
    azure_version = os.environ.get('AZURE_OPENAI_VERSION') or os.environ.get('EMBEDDING_API_VERSION')
    azure_key = os.environ.get('AZURE_OPENAI_API_KEY')
    ollama_host = os.environ.get('OLLAMA_HOST')
    if not (azure_endpoint and azure_version and (azure_key or ollama_host)):
        logger.warning("OPENAI_API_KEY not set. If you are not using Azure or Ollama for embeddings, set OPENAI_API_KEY.")


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8001))

    # Import the app here to ensure environment variables are set first
    from api.api import app

    logger.info(f"Starting Streaming API on port {port}")

    # Run the FastAPI app with uvicorn
    # Disable reload in production/Docker environment
    is_development = os.environ.get("NODE_ENV") != "production"
    
    if is_development:
        # Prevent infinite logging loop caused by file changes triggering log writes
        logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    uvicorn.run(
        "api.api:app",
        host="0.0.0.0",
        port=port,
        reload=is_development
    )
