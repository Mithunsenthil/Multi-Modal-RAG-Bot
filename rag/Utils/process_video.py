import os
import uuid
import cv2
import yt_dlp  
from PIL import Image
import numpy as np
from .model import model, generate_embedding
from pinecone import Pinecone
from dotenv import load_dotenv
import io
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from django.conf import settings
# Load environment variables
load_dotenv()
import time

class VideoProcessor:
    def __init__(self, pinecone_api_key=None, pinecone_index_name=None):
        """Initialize the VideoProcessor with models and database connection"""
        # Initialize the model
        self.llm = model()
        
        # Initialize Pinecone if credentials are provided
        if pinecone_api_key and pinecone_index_name:
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index_name = pinecone_index_name
            self.index = self.pc.Index(pinecone_index_name)
        else:
            self.pc = None
            self.index = None
        
        # Create video directory if it doesn't exist
        self.video_dir = os.path.join(settings.MEDIA_ROOT, "videos")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
            
        # Create frames directory if it doesn't exist
        self.frames_dir = os.path.join(self.video_dir, "frames")
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

    def download_youtube_video(self, youtube_url):
        """Download a YouTube video and return the local file path"""
        try:
            # Generate a unique video filename
            video_filename = f"youtube_{uuid.uuid4().hex[:8]}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': video_path,
                'quiet': False,
                'no_warnings': False,
            }
            
            # Download the video
            print(f"Downloading YouTube video from: {youtube_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                video_title = info.get('title', 'Unknown Title')
            
            print(f"Video downloaded successfully to {video_path}")
            return video_path, video_title
            
        except Exception as e:
            print(f"Error downloading YouTube video: {str(e)}")
            return None, None

    def extract_frames(self, video_path, max_frames=50):
        """Extract frames from the video with exactly max_frames evenly distributed"""
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"Video stats: {total_frames} frames, {fps} FPS, {duration:.2f} seconds")
            
            # Calculate time intervals for exactly max_frames
            time_interval = duration / max_frames  # seconds between each frame
            frames_interval = int(time_interval * fps)  # convert to frame count
            frames_interval = max(1, frames_interval)  # ensure at least 1
            
            print(f"Extracting frames at {time_interval:.2f} second intervals ({frames_interval} frames apart)")
            
            # Extract frames
            frames = []
            frame_paths = []
            
            # Calculate frame positions for exactly max_frames
            frame_positions = []
            for i in range(max_frames):
                # Calculate position as a percentage of video duration
                position_ratio = i / (max_frames - 1) if max_frames > 1 else 0
                frame_position = int(position_ratio * total_frames)
                frame_positions.append(min(frame_position, total_frames - 1))
            
            # Extract the frames at calculated positions
            for i, position in enumerate(frame_positions):
                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB (cv2 uses BGR by default)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Save frame
                    frame_path = os.path.join(self.frames_dir, f"frame_{i:04d}.jpg")
                    pil_image.save(frame_path)
                    
                    frames.append(pil_image)
                    frame_paths.append(frame_path)
            
            cap.release()
            
            print(f"Extracted exactly {len(frames)} frames evenly distributed across {duration:.2f} seconds")
            return frames, frame_paths
            
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return [], []

    def store_in_pinecone(self, frame_data, video_path, video_title, transcript=""):
        """Store frame summaries, embeddings and transcript in Pinecone"""
        try:
            # Generate a unique ID for the video
            video_id = str(uuid.uuid4())
            
            # Get relative path for storage in metadata
            if os.path.isabs(video_path):
                # Calculate relative path for absolute paths
                if video_path.startswith(os.path.join(os.getcwd(), 'media')):
                    relative_path = os.path.relpath(video_path, os.path.join(os.getcwd(), 'media'))
                else:
                    # If not under media, just use the filename
                    relative_path = os.path.basename(video_path)
            else:
                # Already relative path
                relative_path = video_path
                
            # Store each frame
            vectors = []
            for frame in frame_data:
                # Generate a unique ID for the frame
                frame_id = f"{video_id}_frame_{frame['frame_number']:04d}"
                
                # Calculate timestamp if possible
                frame_number = frame['frame_number']
                total_frames = len(frame_data)
                # Assuming a standard 30fps video if no other information
                timestamp = (frame_number / total_frames) * 180  # Assume 3-minute video by default
                
                # Create metadata
                metadata = {
                    "text": frame["summary"],
                    "type": "video_frame",
                    "video_title": video_title,
                    "video_relative_path": relative_path,
                    "frame_number": frame["frame_number"],
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "transcript_snippet": transcript[:1000] if transcript else ""
                }
                
                # Create vector for Pinecone
                vector = {
                    "id": frame_id,
                    "values": frame["embedding"],
                    "metadata": metadata
                }
                vectors.append(vector)
            
            # Upsert vectors in batches
            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                try:
                    self.index.upsert(vectors=batch)
                    print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
                except Exception as e:
                    print(f"Error upserting batch: {str(e)}")
            
            print(f"Successfully processed and stored {len(vectors)} frames from video '{video_title}'")
            return vectors
            
        except Exception as e:
            print(f"Error storing in Pinecone: {str(e)}")
            return None

    def extract_audio(self, video_path):
        """Extract audio from video file"""
        try:
            audio_path = os.path.join(self.video_dir, "temp_audio.wav")
            clip = VideoFileClip(video_path)
            audio = clip.audio
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return None

    def transcribe_audio(self, audio_path):
        """Convert audio to text using whisper"""
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_whisper(audio_data)
                return text
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return ""

    def process_video(self, input_path, max_frames=30):
        """Process a video file or YouTube URL
        
        Args:
            input_path: Either a local file path or YouTube URL
            max_frames: Maximum number of frames to extract
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            # Check if input is a YouTube URL
            is_youtube_url = input_path.startswith(('http://', 'https://', 'www.youtube', 'youtube.com', 'youtu.be'))
            
            if is_youtube_url:
                # Download the YouTube video
                print(f"Detected YouTube URL: {input_path}")
                video_path, video_title = self.download_youtube_video(input_path)
                if not video_path:
                    return False
            else:
                # Use the local file path
                if not os.path.exists(input_path):
                    print(f"Error: File not found: {input_path}")
                    return False
                    
                video_path = input_path
                # Extract video title from filename
                video_title = os.path.basename(input_path).split('.')[0]
                print(f"Processing local video file: {video_path}")
            
            # Extract audio and transcribe
            audio_path = self.extract_audio(video_path)
            transcript = ""
            if audio_path:
                transcript = self.transcribe_audio(audio_path)
                print(f"Generated transcript of {len(transcript)} characters")
                # Clean up temp audio file
                os.remove(audio_path)
            
            # Extract frames
            frames, frame_paths = self.extract_frames(video_path, max_frames)
            if not frames:
                return False
            
            # Process frames with added transcript context
            frame_data = self.process_frames_with_transcript(frames, video_title, transcript)
            if not frame_data:
                return False
            
            # Store in Pinecone
            video_id = self.store_in_pinecone(frame_data, video_path, video_title, transcript)
            if not video_id:
                return False
            
            print(f"Successfully processed video: {video_title}")
            print(f"Video ID: {video_id}")
            return True
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False

    def process_frames_with_transcript(self, frames, video_title, transcript):
        """Process frames with transcript for better context"""
        try:
            frame_data = []
            
            # Create a condensed transcript if it's too long
            condensed_transcript = transcript[:500] + "..." if len(transcript) > 500 else transcript
            
            for i, frame in enumerate(frames):
                # Generate summary for the frame with transcript context
                prompt = f"""Describe this video frame in detail. 
                Focus on key visual elements, actions, and context.
                Consider this transcript from the video: '{condensed_transcript}'
                dont include any markdown , the output is just plain text in lower form , as it will be converted to vectorembedding"""

                # Get summary from Gemini model
                response = self.llm.generate_content([frame, prompt])
                frame_summary = response.text 

                if i%10==0: #added 50 sec delay to avoid rate limit error
                    time.sleep(50)
                
                # Add frame number and video context
                full_summary = f"Frame {i+1} from video '{video_title}': {frame_summary}"
                print(f"Frame {i+1}/{len(frames)}: {frame_summary[:100]}...")
                
                # Generate embedding for the summary
                embedding = generate_embedding(full_summary)
                
                # Add to frame data
                frame_data.append({
                    "frame_number": i+1,
                    "summary": full_summary,
                    "embedding": embedding
                })
                
            return frame_data
            
        except Exception as e:
            print(f"Error processing frames with transcript: {str(e)}")
            return []

#     def search_video_frames(self, query, top_k=2):
#         """Search for video frames based on a text query"""
#         try:
#             # Generate query embedding
#             query_embedding = generate_embedding(query)
            
#             # Search in Pinecone
#             search_results = self.index.query(
#                 vector=query_embedding,
#                 top_k=top_k,
#                 include_metadata=True,
#                 filter={"type": "video_frame"}
#             )
            
#             # Process results
#             results = []
#             for match in search_results.matches:
#                 results.append({
#                     "score": match.score,
#                     "frame_summary": match.metadata.get("text", ""),
#                     "video_title": match.metadata.get("video_title", ""),
#                     "frame_number": match.metadata.get("frame_number", 0),
#                     "video_id": match.metadata.get("video_id", "")
#                 })
            
#             return results
            
#         except Exception as e:
#             print(f"Error searching video frames: {str(e)}")
#             return []

# # Example usage
# if __name__ == "__main__":
#     processor = VideoProcessor()
    
#     # Example with YouTube URL
#     # youtube_url = "https://youtu.be/3dhcmeOTZ_Q"
#     # processor.process_video(youtube_url, max_frames=30)
    
#     # Example with local file path (uncomment and modify with your file path)
#     # local_file = "path/to/your/video.mp4"
#     # processor.process_video(local_file, max_frames=30)
    
#     # Optional: Search for frames
#     search_query = "Explain what is linear regression"
#     if search_query:
#         results = processor.search_video_frames(search_query)
#         for i, result in enumerate(results):
#             print(f"\nResult {i+1} (Score: {result['score']:.4f})")
#             print(f"Video: {result['video_title']}")
#             print(f"Frame: {result['frame_number']}")
#             print(f"Summary: {result['frame_summary']}")
