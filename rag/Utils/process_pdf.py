import fitz  # PyMuPDF
import pdfplumber
import re
from PIL import Image
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
from .model import generate_embedding, model
import base64
from io import BytesIO

class PDFProcessor:
    def __init__(self, pinecone_api_key, pinecone_index_name, chunk_size=1000, chunk_overlap=300):
        # Initialize Pinecone if credentials are provided
        if pinecone_api_key and pinecone_index_name:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index_name = pinecone_index_name

            # Check if index exists, create if not
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Match the embedding model's output dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-west-1'  # Correct region for free tier
                    )
                )
            self.index = self.pc.Index(self.index_name)
            print("index created")
        else:
            self.pc = None
            self.index = None

        # Initialize text splitting parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", r"(?<!\d)\.(?!\d)", " ", ""]
        
        # Initialize Google model
        self.google_model = model()

    def _split_text(self, text):
        """Internal text splitting method"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            separator = self.separators[0]
            for sep in self.separators:
                if sep == "":
                    split_point = end
                    break
                matches = list(re.finditer(sep, text[start:end], flags=re.MULTILINE))
                if matches:
                    last_match = matches[-1]
                    split_point = start + last_match.end()
                    separator = sep
                    break

            chunk = text[start:split_point]
            chunks.append(chunk)
            start = split_point - self.chunk_overlap if split_point - self.chunk_overlap > start else split_point

        return chunks

    def extract_content(self, pdf_path):
        """Extract text, tables, and images from PDF"""
        text = ""
        tables = []
        images = []

        # Extract text and images
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:  # Ensure text is not None
                    text += page_text
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image", None)
                    if image_bytes:
                        try:
                            # Create a directory if it doesn't exist
                            os.makedirs("extracted_images", exist_ok=True)
                            
                            # Save the image locally
                            image_path = f"extracted_images/page_{page_num}_image_{img_index}.png"
                            with open(image_path, "wb") as image_file:
                                image_file.write(image_bytes)
                            images.append(image_path)  # Store the local path of the image
                        except Exception as e:
                            print(f"Error saving image: {e}")

        # Extract tables
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:  # Ensure tables are not None
                    tables.extend(page_tables)

        return text, tables, images

    def process_pdf(self, pdf_path):
        """Full PDF processing pipeline"""
        text, tables, images = self.extract_content(pdf_path)

        # Prepare image chunks with local paths
        image_chunks = []
        for i, image_path in enumerate(images):
            # Generate caption
            image = Image.open(image_path)
            prompt = "You are an assistant tasked with summarizing images for retrieval. These summaries will be embedded and used to retrieve the raw image. Give a concise summary of the image that is well optimized for retrieval. Don't have a markdown or a heading summary, output must only have the summary in the lower text form."
            response = self.google_model.generate_content([image, prompt])
            caption = response.text
            # Add image chunk with metadata
            image_chunks.append({
                "id": f"image_{i}",
                "text": caption,
                "type": "image",
                "image_path": image_path  # Store the local path of the image
            })

        # Prepare table chunks
        table_chunks = []
        for i, table in enumerate(tables):
            if table:
                table_text = ""
                for row in table:
                    if row:
                        cleaned_row = [str(cell) if cell is not None else "" for cell in row]
                        table_text += "|".join(cleaned_row) + "\n"
                table_chunks.append({"id": f"table_{i}", "text": table_text, "type": "table"})

        # Prepare text chunks
        text_chunks = self._split_text(text)
        text_chunks = [{
            "id": f"text_{i}", 
            "text": chunk, 
            "type": "text"
        } 
        for i, chunk in enumerate(text_chunks)]

        # Combine all chunks
        all_chunks = text_chunks + table_chunks + image_chunks

        # Debug: Print the number of chunks
        print(f"Number of chunks: {len(all_chunks)} (Text: {len(text_chunks)}, Tables: {len(table_chunks)}, Images: {len(image_chunks)})")

        # Prepare vectors for Pinecone
        vectors = []
        for i, chunk in enumerate(all_chunks):
            # Generate embedding for the chunk text
            embedding = generate_embedding(chunk["text"])
            
            metadata = {
                "text": chunk["text"],
                "type": chunk["type"]
            }
            
            # Add image path to metadata if the chunk is an image
            if chunk["type"] == "image" and "image_path" in chunk:
                metadata["image_path"] = chunk["image_path"]
            
            # Create vector
            vector = {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": metadata
            }
            
            vectors.append(vector)

        # Debug: Print vector details
        print(f"Number of vectors to be uploaded: {len(vectors)}")
        print(f"Example vector metadata: {vectors[0]['metadata']}")

        # Upsert vectors to Pinecone
        for i in range(0, len(vectors), 50):
            batch = vectors[i:i+50]
            print(f"Uploading batch {i//100 + 1} with {len(batch)} vectors")
            try:
                response = self.index.upsert(vectors=batch)
                print(f"Batch {i//100 + 1} uploaded successfully")
            except Exception as e:
                print(f"Error upserting batch {i//100 + 1}: {str(e)}")

        return all_chunks
