import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Application Configuration
    """

    # -------------------------
    # OpenAI
    # -------------------------

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    CHAT_MODEL = "gpt-5-mini"
    API_BASE_URL = "http://127.0.0.1:8000"

    EMBEDDING_MODEL = "text-embedding-3-small"

    # -------------------------
    # Local Embedding Model
    # -------------------------

    LOCAL_EMBEDDING_MODEL = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------
    # Chunk Settings
    # -------------------------

    CHUNK_SIZE =200 #500
    CHUNK_OVERLAP =80 #100

    # -------------------------
    # Vector DB
    # -------------------------

    CHROMA_DIRECTORY = "./chroma_db"

    # -------------------------
    # Retrieval
    # -------------------------

    TOP_K_RESULTS = 3
    #Database connection settings
    # DB_HOST = os.getenv("DB_HOST")
    # DB_PORT = os.getenv("DB_PORT")
    # DB_NAME = os.getenv("DB_NAME")
    # DB_USER = os.getenv("DB_USER")
    # DB_PASSWORD = os.getenv("DB_PASSWORD")