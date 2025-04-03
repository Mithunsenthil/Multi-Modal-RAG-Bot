# RAG-based Media Processing System

## 1. Introduction

This Retrieval-Augmented Generation (RAG) Chatbot uses Django framework to build a full-stack web application.
Implemented a MultiModal RAG system,  that queries from different document types like pdf, docs, csv, image and video, used different processing techniques for different document types.
Knowledge Base:
    Students Grading Dataset [CSV File]
    Data Incident Management [PDF File]
    Luxury Cars Image Dataset [Image Files]
    Linear Regression in 3 Minutes [Video File]
    Introduction to Machine Learning [DOC File]
The user can also upload their own data into the system and can be queried from that also.


Insert file Processing

Each file type in handled differently to extract the context. Every file processing module is written as a class for modularity.
PDF File handling:
Main function extract_content() and process_pdf()
The extract_content() read the pdf and extracts all the text, tables and images.
The process_pdf() chunks the content in the file each image and tables is a separate chunk and the text is splited using _split_text() function with chunk_size=1000, chunk_overlap=300
Finally all these chunks are embedded upserted along with some metadata into vector DB (pinecone)


User Query Processing

This is handled by rag_bot.py [rag\Utils\rag_bot.py]
The RAGbot class has two methods retrieve_context() and normal_response()
The retrieve_context() will take the user query as its input, the user query is embedding and searched in pinecone for all the related context and the function returns top 4 the matches. 
The normal_response() will take the user query and matches from the db and extracts all the context from the matches and feeds it to the llama3-70b-8192 used via groq API to generate response for the user query. Used prompt engineering for optimal response.


Csv Query Processing

If a csv content is present in the related context for the user query it will be processed differently.
The summary of the csv along with the user query will be given to a LLM which will produce a pandas code for the user query, which will be executed and the pandas output is stored.
The pandas code, pandas output along, csv summary along with the user query will be given to another LLM which will give a final response to the user query.


Insert file Processing

Each file type in handled differently to extract the context. Every file processing module is written as a class for modularity.

PDF File handling:
This is handled by process_pdf.py [rag\Utils\process_pdf.py]
Main function extract_content() and process_pdf()
The extract_content() read the pdf and extracts all the text, tables and images.
The process_pdf() chunks the content in the file each image and tables is a separate chunk and the text is splited using _split_text() function with chunk_size=1000, chunk_overlap=300
Finally all these chunks are embedded upserted along with some metadata into vector DB (pinecone)


Docx File handling:
This is handled by process_doc.py [rag\Utils\process_doc.py]
Main function extract_content() and process_documents()
The extract_content() read the Doc or txt file and extracts all the text, tables.
The process_documents() chunks the content in the file each image and tables is a separate chunk and the text is splited using _split_text() function with chunk_size=1000, chunk_overlap=200
Finally all these chunks are embedded upserted along with some metadata into vector DB (pinecone)

Csv File handling:
This is handled by process_csv.py [rag\Utils\process_csv.py]
Main function generate_csv_summary() and process_csv()
The generate_csv_summary() read the csv file and extracts all the important information.  
The process_csv() create embedding for the csv summary and it is upserted into the vector DB (pinecone)


Image File handling:
This is handled by process_image.py [rag\Utils\process_image.py]
Main function generate_image_summary() and process_image()
The optimize_image() is used to reduce the size of the image for storing, the generate_image_summary() is used to create a summary for the image using the gemini-2.0-flash which is the optimal model for image summarization.
The image summary is embedded and upserted in the vector DB along with the metadata which contains the base64 encoding of the image file. This base64 encoding will be later used for displaying the image when required. 




Video File handling:
This is handled by process_video.py [rag\Utils\process_video.py]
The main functions process_video() and process_frames_with_transcript() handle video processing. 
The extract_frames() is used to extract key frames from the video, while transcribe_audio() is used to extract audio transcribe and these both are given to  process_frames_with_transcript() which creates context with gemini-2.0-flash using both audio and frames. 
Finally all these are embedded upserted along with some metadata into vector DB (pinecone).

## Prerequisites

```python
# Required Python packages
opencv-python
yt-dlp
Pillow
numpy
pinecone-client
python-dotenv
SpeechRecognition
moviepy
```

## Environment Setup

1. Create a `.env` file with your API keys:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```

### Storage
- Pinecone vector database
- Batch processing for efficiency
- Metadata includes:
  - Frame/Image summaries
  - File paths
  - Timestamps (for videos)
  - Transcripts (for videos)
  - Base64 encoded images

## Rate Limiting
- 50-second delay every 10 frames for API compliance
- Batch processing for vector storage

## Error Handling
- Comprehensive try-except blocks
- Detailed error logging
- Graceful failure handling

## 3. Tech Stack
### Backend
- Language: Python
- Technology: RAG
- Embedding model: text-embedding-004
- image model: gemini-2.0-flash
- Chat model: Llama3-70b-8192
- image processing: OpenCV
- Vector DB: Pinecone Db

- Note: No frameworks like langchian is used.


### Storage
- Pinecone (Vector Database)


## 4. Setup and Installation

### Prerequisites
- Python 3.8+
- Required API Keys:
  - Pinecone
  - Gemini AI
  - Groq 

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Create `.env` file:
```env
# Django Configuration
DJANGO_SECRET_KEY=your_django_secret_key
DEBUG=True

# API Keys
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key


### Running the Application

1. Apply migrations:
```bash
python manage.py migrate
```

2. Start development server:
```bash
python manage.py runserver
```


## URL Endpoints
- `/chat/` - Chat interface for interacting with the system
- `/user/home/` - Home page 

### Media Upload
- `/chat/` - Chat interface for interacting with the system
- `/upload-pdf/` - PDF file upload and processing
- `/upload-doc/` - DOC/DOCX file upload and processing  
- `/upload-csv/` - CSV file upload and processing
- `/upload-video/` - Video file upload and processing
- `/upload-image/` - Image file upload and processing



## Error Handling
- Comprehensive try-except blocks
- Detailed error logging
- Rate limiting for API calls
- Batch processing for large operations

