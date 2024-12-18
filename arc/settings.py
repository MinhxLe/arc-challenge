import os
from dotenv import load_dotenv

load_dotenv()

OAI_API_KEY = os.getenv("OAI_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

TEMP_ROOT_DIR = "./tmp"
