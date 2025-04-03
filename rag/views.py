from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .Utils.process_pdf import PDFProcessor
from .Utils.rag_bot import RAGBot
from .Utils.process_doc import DocumentProcessor
from .Utils.process_csv import CSVProcessor
from .Utils.process_video import VideoProcessor
from .Utils.process_image import ImageProcessor
from .Utils.model import generate_embedding
import os
from django.core.files.storage import default_storage
from django.conf import settings
import pinecone
import base64


# Configuration (move these to Django settings.py for production)
PINECONE_CONFIG = {
    "api_key": os.getenv("PINECONE_API_KEY"),
    "index": os.getenv("PINECONE_INDEX_NAME")
}
GROQ_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_KEY)
print(PINECONE_CONFIG)
# PDF_PATH = "Data/Data Science Interview.pdf"

# Initialize the RAG system (do this once, not on every request)
# processor = PDFProcessor(
#     pinecone_api_key=PINECONE_CONFIG["api_key"],
#     pinecone_index_name=PINECONE_CONFIG["index"]
# )
# processor.process_pdf(PDF_PATH)  # Process the PDF and upload embeddings to Pinecone

bot = RAGBot(
    groq_api_key=GROQ_KEY,
    pinecone_api_key=PINECONE_CONFIG["api_key"],
    pinecone_index_name=PINECONE_CONFIG["index"]
)


@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')

        # Retrieve relevant context from Pinecone
        context = bot.retrieve_context(user_message)
        print("context",context)
        # Generate response using appropriate model based on query type
        response, img_b64, full_context = bot.normal_response(user_message, context)
        print("response",response)
        # Prepare the response payload
        payload = {'message': response.strip(), 'related': []}
        chunk_data ={
            'image':img_b64,
            'text':full_context
        }
        payload['related'].append(chunk_data)

        return JsonResponse(payload)

    return render(request, 'chat.html')

@csrf_exempt
def upload_pdf(request):
    processor = PDFProcessor(
        pinecone_api_key=PINECONE_CONFIG["api_key"],
        pinecone_index_name=PINECONE_CONFIG["index"]
    )
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        # Save the uploaded file temporarily
        uploaded_file = request.FILES['pdf_file']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        try:
            # Process the PDF and upload embeddings to Pinecone
            chunks = processor.process_pdf(file_path)
            
            # Clean up the temporary file
            # if default_storage.exists(file_path):
            #     default_storage.delete(file_path)
            
            # Render the success message
            return render(request, 'upload_pdf.html', {
                'status': 'success',
                'message': f"PDF processed successfully. {len(chunks)} chunks uploaded to Pinecone."
            })
        except Exception as e:
            # Clean up the temporary file in case of error
            # if default_storage.exists(file_path):
            #     default_storage.delete(file_path)
            
            # Render the error message
            return render(request, 'upload_pdf.html', {
                'status': 'error',
                'message': str(e)
            })
    
    # Render the upload form
    return render(request, 'upload_pdf.html')

@csrf_exempt
def upload_doc(request):
    processor = DocumentProcessor(
        pinecone_api_key=PINECONE_CONFIG["api_key"],
        pinecone_index_name=PINECONE_CONFIG["index"]
    )
        
    if request.method == 'POST' and request.FILES.get('doc_file'):
        # Save the uploaded file temporarily
        uploaded_file = request.FILES['doc_file']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        try:
            # Process the DOCX and upload embeddings to Pinecone
            chunks = processor.process_documents(file_path)  # Using the existing method from the processor
            
            # Clean up the temporary file
            # if default_storage.exists(file_path):
            #     default_storage.delete(file_path)
            
            # Render the success message
            return render(request, 'upload_doc.html', {
                'status': 'success',
                'message': f"DOCX processed successfully. {len(chunks)} chunks uploaded to Pinecone."
            })
        except Exception as e:
            # Clean up the temporary file in case of error
            if default_storage.exists(file_path):
                default_storage.delete(file_path)
            
            # Render the error message
            return render(request, 'upload_doc.html', {
                'status': 'error',
                'message': str(e)
            })
    
    # Render the upload form
    return render(request, 'upload_doc.html')


@csrf_exempt
def upload_csv(request):

    processor = CSVProcessor(
        pinecone_api_key=PINECONE_CONFIG["api_key"],
        pinecone_index_name=PINECONE_CONFIG["index"],
        groq_api_key=GROQ_KEY
    )

    if request.method == 'POST' and request.FILES.get('csv_file'):
        # Save the uploaded file temporarily
        uploaded_file = request.FILES['csv_file']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            # Process the CSV and upload embeddings to Pinecone
            processor.process_csv(file_path)

            # Clean up the temporary file
            # if default_storage.exists(file_path):
            #     default_storage.delete(file_path)

            # Render the success message
            return render(request, 'upload_csv.html', {
                'status': 'success',
                'message': f"CSV processed successfully and uploaded to Pinecone."
            })
        except Exception as e:
            # Clean up the temporary file in case of error
            if default_storage.exists(file_path):
                default_storage.delete(file_path)

            # Render the error message
            return render(request, 'upload_csv.html', {
                'status': 'error',
                'message': str(e)
            })

    # Render the upload form
    return render(request, 'upload_csv.html')

@csrf_exempt
def upload_video(request):
    processor = VideoProcessor(
        pinecone_api_key=PINECONE_CONFIG["api_key"],
        pinecone_index_name=PINECONE_CONFIG["index"]
    )
        
    if request.method == 'POST':
        # Handle YouTube URL upload
        youtube_url = request.POST.get('youtube_url', '')
        
        if youtube_url:
            # Process YouTube URL
            try:
                # Process the video from YouTube URL and upload embeddings to Pinecone
                vectors = processor.process_video(youtube_url)
                
                if not vectors:
                    return render(request, 'upload_video.html', {
                        'status': 'error',
                        'message': "Failed to process YouTube video."
                    })
                
                # Render the success message
                return render(request, 'upload_video.html', {
                    'status': 'success',
                    'message': f"YouTube video processed successfully and uploaded to Pinecone."
                })
            except Exception as e:
                return render(request, 'upload_video.html', {
                    'status': 'error',
                    'message': str(e)
                })
        
        # Handle file upload
        elif request.FILES.get('video_file'):
            uploaded_file = request.FILES['video_file']
            
            # Create a videos subdirectory in the media folder if it doesn't exist
            videos_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
            os.makedirs(videos_dir, exist_ok=True)
            
            # Save the video file to the videos directory
            file_path = os.path.join(videos_dir, uploaded_file.name)
            
            with default_storage.open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            try:
                # Process the video and upload embeddings to Pinecone
                vectors = processor.process_video(file_path)
                
                if not vectors:
                    return render(request, 'upload_video.html', {
                        'status': 'error',
                        'message': "Failed to process video file."
                    })
                
                # Calculate the relative URL for display
                relative_path = os.path.join('videos', uploaded_file.name)
                video_url = f"{settings.MEDIA_URL}{relative_path}"
                
                # Render the success message
                return render(request, 'upload_video.html', {
                    'status': 'success',
                    'message': f"Video processed successfully and uploaded to Pinecone.",
                    'video_path': video_url
                })
            except Exception as e:
                # On error, we'll keep the video but report the error
                return render(request, 'upload_video.html', {
                    'status': 'error',
                    'message': str(e),
                    'video_path': None
                })
        else:
            # Neither URL nor file provided
            return render(request, 'upload_video.html', {
                'status': 'error',
                'message': "Please provide either a YouTube URL or upload a video file."
            })
    
    # Render the upload form
    return render(request, 'upload_video.html')

@csrf_exempt
def upload_image(request):
    processor = ImageProcessor(
        pinecone_api_key=PINECONE_CONFIG["api_key"],
        pinecone_index_name=PINECONE_CONFIG["index"]
    )
        
    if request.method == 'POST' and request.FILES.get('image_file'):
        # Save the uploaded file temporarily
        uploaded_file = request.FILES['image_file']
        
        # Create an images subdirectory in the media folder if it doesn't exist
        images_dir = os.path.join(settings.MEDIA_ROOT, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save the image file to the images directory
        file_path = os.path.join(images_dir, uploaded_file.name)
        
        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        
        try:
            # Process the image and upload embeddings to Pinecone
            result = processor.process_image(file_path)
            
            if not result:
                return render(request, 'upload_image.html', {
                    'status': 'error',
                    'message': "Failed to process the image."
                })
            
            # Calculate the relative URL for display
            relative_path = os.path.join('images', uploaded_file.name)
            image_url = f"{settings.MEDIA_URL}{relative_path}"
            
            # Render the success message
            return render(request, 'upload_image.html', {
                'status': 'success',
                'message': f"Image processed successfully. Summary: {result.get('summary', '')}",
                'image_path': image_url
            })
        except Exception as e:
            # On error, we'll keep the image but report the error
            return render(request, 'upload_image.html', {
                'status': 'error',
                'message': str(e),
                'image_path': None
            })
    
    # Render the upload form
    return render(request, 'upload_image.html')