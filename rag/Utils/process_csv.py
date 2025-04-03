import pandas as pd
import os
import uuid
from pinecone import Pinecone
from groq import Groq
from .model import generate_embedding

class CSVProcessor:
    def __init__(self, pinecone_api_key=None, pinecone_index_name=None, groq_api_key=None):
        """Initialize the CSV processor with Pinecone and Groq credentials."""
        # Initialize Pinecone if credentials are provided
        if pinecone_api_key and pinecone_index_name:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(pinecone_index_name)
        else:
            self.pc = None
            self.index = None
            
        # Initialize Groq if API key is provided
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            self.groq_client = None

    def process_csv(self, file_path):
        """Process a CSV file and store in Pinecone with embeddings."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        # Read CSV file
        try:
            df = pd.read_csv(file_path)
            print(f"CSV file loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
            
        # Generate summary of the CSV
        summary = self.generate_csv_summary(df, file_path)
        print(f"Generated CSV summary: {summary[:100]}...")
        
        # Create embedding for the summary
        embedding = generate_embedding(summary)
        
        # Create vector for Pinecone
        vector_id = str(uuid.uuid4())
        vector = {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": summary,
                "type": "csv",
                "file_path": file_path,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }
        }
        
        # Upsert to Pinecone if available
        if self.index:
            self.index.upsert(vectors=[vector])
            print(f"Uploaded CSV summary to Pinecone with ID: {vector_id}")
            
        return [vector]
        
    def generate_csv_summary(self, df, file_path):
        """Generate a summary of the CSV using Groq or basic statistics."""
        # Collect basic statistics
        file_name = os.path.basename(file_path)
        num_rows = len(df)
        num_cols = len(df.columns)
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Get sample data (first 5 rows)
        sample = df.head(5).to_string()
        
        # Build summary text
        summary = f"CSV File: {file_name}\n"
        summary += f"Rows: {num_rows}, Columns: {num_cols}\n"
        summary += "Column Names and Types:\n"
        
        for col, dtype in column_types.items():
            summary += f"- {col} ({dtype})\n"
            
        summary += f"\nSample Data:\n{sample}\n"
        
        # Use Groq to generate a more detailed summary if available
        if self.groq_client:
            try:
                prompt = f"""Analyze this CSV data and create a concise summary:
                
                {summary}
                
                Describe the data in detail, including potential use cases. Focus on what questions could be answered with this dataset.
                """
                
                response = self.groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a data analyst assistant. Summarize CSV datasets concisely."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=500
                )
                
                detailed_summary = response.choices[0].message.content
                summary += f"\nAnalysis:\n{detailed_summary}"
            except Exception as e:
                print(f"Error generating detailed summary with Groq: {str(e)}")
        
        return summary