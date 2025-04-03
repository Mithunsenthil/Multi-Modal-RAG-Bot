# from pinecone import Pinecone, ServerlessSpec
# import os
# from dotenv import load_dotenv
# from exception import customexception
# import sys
# load_dotenv()

# def connect_pineconedb():
#     try:
#         pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         index = pc.Index("mithun")
#         print("true for index")
#         return index
#     except Exception as e:
#         raise customexception(e,sys)