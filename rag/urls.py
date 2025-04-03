from django.urls import path
from .views import chat_view, upload_pdf, upload_doc, upload_csv, upload_video, upload_image

urlpatterns = [
    path('chat/', chat_view, name='chat'),
    path('upload-pdf/', upload_pdf, name='upload_pdf'),
    path('upload-doc/', upload_doc, name='upload_doc'),
    path('upload-csv/', upload_csv, name='upload_csv'),
    path('upload-video/', upload_video, name='upload_video'),
    path('upload-image/', upload_image, name='upload_image'),
]
