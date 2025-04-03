import os
import io
import base64
from PIL import Image
from .model import generate_embedding, model 
from pinecone import Pinecone
import uuid
from dotenv import load_dotenv
from io import BytesIO

class ImageProcessor:
    def __init__(self, pinecone_api_key=None, pinecone_index_name=None):
        """Initialize the ImageProcessor with Pinecone credentials."""
        load_dotenv()
        
        # Initialize Pinecone if credentials are provided
        if pinecone_api_key and pinecone_index_name:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index_name = pinecone_index_name
            self.index = self.pc.Index(self.index_name)
        else:
            self.pc = None
            self.index = None
        
        # Initialize the model
        self.llm = model()

    def optimize_image(self, image_path, max_size=(300, 300), quality=50):
        """Optimize an image to fit within Pinecone's metadata size limits"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize the image
                img.thumbnail(max_size)
                
                # Save with compression
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                
                # Get base64 of optimized image
                optimized_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Check size and further compress if needed
                if len(optimized_b64) > 35000:  # Keep under 40KB with some margin
                    # Try again with more aggressive settings
                    return self.optimize_image(image_path, (200, 200), 30)
                
                print(f"Optimized image size: {len(optimized_b64)} bytes")
                return optimized_b64
        except Exception as e:
            print(f"Error optimizing image: {str(e)}")
            return None

    def chunk_image(self, image_data):
        """Split base64 image into multiple chunks if too large"""
        # Maximum size for each chunk (35KB to be safe)
        max_chunk_size = 37000
        
        # If the image is small enough, return it as a single chunk
        if len(image_data) <= max_chunk_size:
            return [image_data]
        
        # Otherwise, split it into chunks
        chunks = []
        for i in range(0, len(image_data), max_chunk_size):
            chunks.append(image_data[i:i + max_chunk_size])
        
        print(f"Split image into {len(chunks)} chunks")
        return chunks

    def generate_image_summary(self, image_path):
        """Generate a summary of the image content using the model"""
        try:
            # Open image for model processing
            image = Image.open(image_path)
            
            # Generate image summary
            prompt = "You are an assistant tasked with summarizing images for retrieval. These summaries will be embedded and used to retrieve the raw image. Give a concise summary of the image that is well optimized for retrieval. Don't have a markdown or a heading summary, output must only have the summary in the lower text form."
            response = self.llm.generate_content([image, prompt])
            return response.text
        except Exception as e:
            print(f"Error generating image summary: {str(e)}")
            # Fallback: use filename as summary
            return f"Image file: {os.path.basename(image_path)}"

    def process_image(self, image_path):
        """Process a single image and store it in Pinecone"""
        try:
            # Try to optimize the image first
            optimized_b64 = self.optimize_image(image_path)
            if not optimized_b64:
                print(f"Failed to optimize image {image_path}, skipping")
                return None
            
            # Generate image summary
            image_summary = self.generate_image_summary(image_path)
            print(f"Image summary: {image_summary}")
            
            # Generate embedding from the summary
            embedding = generate_embedding(image_summary)
            
            # Generate a unique ID for the image
            image_id = str(uuid.uuid4())
            filename = os.path.basename(image_path)
            
            # Check if we need to chunk the image
            b64_chunks = self.chunk_image(optimized_b64)
            
            if len(b64_chunks) == 1:
                # Image fits in a single vector
                metadata = {
                    "b64": optimized_b64,
                    "text": image_summary,
                    "type": "image",
                    "filename": filename
                }
                
                try:
                    vector = {
                        "id": image_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                    self.index.upsert([vector])
                    print(f"Uploaded single vector: {image_id}")
                    return {"id": image_id, "chunks": 1, "summary": image_summary}
                except Exception as e:
                    print(f"Error upserting vector {image_id}: {str(e)}")
                    return None
            else:
                # Image needs multiple chunks
                vectors = []
                for i, chunk in enumerate(b64_chunks):
                    chunk_id = f"{image_id}_chunk_{i}"
                    
                    # First chunk contains the embedding and summary
                    if i == 0:
                        metadata = {
                            "b64_chunk": chunk,
                            "text": image_summary,
                            "type": "image_chunk",
                            "filename": filename,
                            "chunk_index": i,
                            "total_chunks": len(b64_chunks),
                            "parent_id": image_id
                        }
                    else:
                        # Other chunks just store their part of the image data
                        metadata = {
                            "b64_chunk": chunk,
                            "type": "image_chunk",
                            "chunk_index": i,
                            "total_chunks": len(b64_chunks),
                            "parent_id": image_id
                        }
                    
                    vector = {
                        "id": chunk_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                    vectors.append(vector)
                
                try:
                    self.index.upsert(vectors)
                    print(f"Uploaded {len(vectors)} chunks for image {image_id}")
                    return {"id": image_id, "chunks": len(vectors), "summary": image_summary}
                except Exception as e:
                    print(f"Error upserting chunks: {str(e)}")
                    return None
                
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    # def process_images(self, image_directory):
    #     """Process all images in a directory"""
    #     if not os.path.exists(image_directory):
    #         print(f"Directory not found: {image_directory}")
    #         return []
        
    #     # Get all image files in the directory
    #     image_extensions = ['.jpg', '.jpeg', '.png']
    #     image_files = [
    #         os.path.join(image_directory, f) for f in os.listdir(image_directory)
    #         if os.path.isfile(os.path.join(image_directory, f)) and 
    #         any(f.lower().endswith(ext) for ext in image_extensions)
    #     ]
        
    #     print(f"Found {len(image_files)} images in {image_directory}")
        
    #     results = []
    #     for image_path in image_files:
    #         result = self.process_image(image_path)
    #         if result:
    #             results.append(result)
        
    #     return results

    # def query_images(self, query_text, top_k=3):
    #     """Query Pinecone for images similar to the query text"""
    #     if not self.index:
    #         raise ValueError("Pinecone index not initialized")
            
    #     try:
    #         # Generate embedding for the query text
    #         query_embedding = generate_embedding(query_text)
            
    #         # Query Pinecone
    #         results = self.index.query(
    #             vector=query_embedding,
    #             top_k=top_k,
    #             include_metadata=True
    #         )
            
    #         image_results = []
    #         for match in results["matches"]:
    #             metadata = match["metadata"]
                
    #             # Handle regular images
    #             if metadata.get("type") == "image":
    #                 image_results.append({
    #                     "score": match["score"],
    #                     "b64_image": metadata.get("b64"),
    #                     "text": metadata.get("text"),
    #                     "filename": metadata.get("filename"),
    #                     "type": "image"
    #                 })
                
    #             # Handle first chunk of chunked images
    #             elif metadata.get("type") == "image_chunk" and metadata.get("chunk_index") == 0:
    #                 # Found the first chunk of an image, need to fetch all chunks
    #                 parent_id = metadata.get("parent_id")
    #                 total_chunks = metadata.get("total_chunks")
                    
    #                 # Start with the first chunk's data
    #                 b64_chunks = [metadata.get("b64_chunk")]
                    
    #                 # Fetch the rest of the chunks
    #                 for i in range(1, total_chunks):
    #                     chunk_id = f"{parent_id}_chunk_{i}"
    #                     try:
    #                         chunk_vector = self.index.fetch(ids=[chunk_id])
    #                         if chunk_id in chunk_vector["vectors"]:
    #                             chunk_metadata = chunk_vector["vectors"][chunk_id]["metadata"]
    #                             b64_chunks.append(chunk_metadata.get("b64_chunk"))
    #                     except Exception as e:
    #                         print(f"Error fetching chunk {i}: {str(e)}")
                    
    #                 # Combine all chunks into one image
    #                 if len(b64_chunks) == total_chunks:
    #                     full_b64 = "".join(b64_chunks)
    #                     image_results.append({
    #                         "score": match["score"],
    #                         "b64_image": full_b64,
    #                         "text": metadata.get("text"),
    #                         "filename": metadata.get("filename"),
    #                         "type": "image_chunk"
    #                     })
            
    #         return image_results
            
    #     except Exception as e:
    #         print(f"Error querying images: {str(e)}")
    #         return []


# def search_images(query, top_k=3):
#     processor = ImageProcessor()
#     return processor.query_images(query, top_k)

# def display_image_results(results):
#     if not results:
#         print("No matching images found.")
#         return
    
#     print(f"Found {len(results)} matching images:")
#     for i, result in enumerate(results):
#         print(f"\nResult {i+1} (Score: {result['score']:.4f})")
#         print(f"Context: {result.get('text', 'No context')}")
        
#         try:
#             # Display image from base64
#             image_data = base64.b64decode(result['b64_image'])
#             image = Image.open(BytesIO(image_data))
#             image.show()
#         except Exception as e:
#             print(f"Error displaying image: {str(e)}")


# if __name__ == "__main__":
#     processor = ImageProcessor()
#     # Test processing a directory
#     # processor.process_images("images")
    
#     # Test querying
#     results = processor.query_images("image of a car")
#     display_image_results(results)
