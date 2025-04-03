import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()


def model():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model= genai.GenerativeModel("gemini-2.0-flash")
    return model

def generate_embedding(text):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    embedding = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return embedding["embedding"]