from groq import Groq
from pinecone import Pinecone
from .process_image import ImageProcessor
from .model import generate_embedding


class RAGBot:
    def __init__(self, groq_api_key, pinecone_api_key, pinecone_index_name):
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        

    def retrieve_context(self, query, top_k=4):
        """Retrieve relevant context from Pinecone, including metadata."""
        # Use generate_embedding for all queries
        query_embedding = generate_embedding(query)
            
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches
    
    def normal_response(self, query, matches):
        """Generate a response using the context matches."""
        
        full_context = ""
        img_b64=[]
        for match in matches:
            if match['metadata']['type'] == 'csv':
                csv_prompt = f"""You are given a CSV summary and a user query. Generate a pandas code snippet to satisfy the query.
                CSV Summary: {match['metadata']['text']}
                User Query: {query}
                Provide only the pandas code:"""

                code_completion = self.groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are a technical expert. Generate only pandas code to satisfy the query. and put the answer in result variable.and dont plot or print anyting"},
                        {"role": "user", "content": csv_prompt}
                    ],
                    temperature=0,
                    max_tokens=512,
                    stop=None
                )
                pandas_code = code_completion.choices[0].message.content.strip()

                # Execute the pandas code
                try:
                    local_vars = {}
                    # Use raw string to handle file paths
                    file_path = match.metadata.get('file_path', '').replace('\\', '/')
                    pandas_code = f"df = pd.read_csv(r'{file_path}')\n"
                    pandas_code += code_completion.choices[0].message.content.strip()
                    exec(pandas_code, {"pd": __import__("pandas")}, local_vars)
                    pandas_output = local_vars.get("result", "No output generated")
                except Exception as e:
                    pandas_output = f"Error executing pandas code: {str(e)}"

                # Combine pandas output with metadata text
                combined_prompt = f"""Context: {match['metadata']['text']}
                Pandas Output: {pandas_output}"""
                full_context += combined_prompt + "\n\n"

                print("\n\npandas_code\n\n",pandas_code)
                print("\n\npandas_output\n\n",pandas_output)
            else:
                full_context += f"{match['metadata']['text']}\n"
                if match['metadata']['type'] == 'image':
                    if 'b64' in match['metadata']:
                        img_b64.append(match['metadata']['b64'])
        final_prompt = f"""Answer based on the following context:
        {full_context}
        if there is  no context , then answer that data is not present in knowlede base dont give any answer wit releated to query if no context is provided"""

        print("final_prompt",final_prompt)
        final_completion = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=512,
            stop=None
        )
        final_response = final_completion.choices[0].message.content.strip()
        print (len(img_b64))
        return final_response , img_b64 , full_context


# # Usage
# if __name__ == "__main__":
#     # Configuration
#     PINECONE_CONFIG = {
#         "api_key": "pcsk_7RjTQN_UamNytpW3VLRmZw2LUKN9dM3FQWrCyJZLS2WBSRSpsZPt2yAENrx8Don1j7STgC",
#         "index": "rag-system"
#     }
#     GROQ_KEY = "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"
#     PDF_PATH = "Data/Data Science Interview.pdf"

#     # Process PDF and upload to Pinecone
#     processor = PDFProcessor(
#         pinecone_api_key=PINECONE_CONFIG["api_key"],
#         pinecone_index_name=PINECONE_CONFIG["index"]
#     )
#     processor.process_pdf(PDF_PATH)

#     # Query the RAG system
#     bot = RAGBot(
#         groq_api_key=GROQ_KEY,
#         pinecone_api_key=PINECONE_CONFIG["api_key"],
#         pinecone_index_name=PINECONE_CONFIG["index"]
#     )
#     query = "Explain the main concepts discussed in the document"
#     context_matches = bot.retrieve_context(query)
#     response, related_chunks = bot.generate_response(query, context_matches, include_related_chunks=True)
#     print("Response:", response)
#     print("Related Chunks:", related_chunks)
