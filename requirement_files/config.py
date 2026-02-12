from dotenv import load_dotenv
import os 

load_dotenv()

API_KEY = os.getenv("COURIER_API_KEY")
USER_EMAIL = os.getenv("EMAIL")
