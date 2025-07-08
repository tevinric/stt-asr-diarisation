from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import librosa
import io
import tempfile
import os
from typing import List, Dict, Any
import asyncio
import uuid
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
whisper_model = None
diarization_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global whisper_model, diarization_pipeline
    
    logger.info("Loading models...")
    
    # Load Whisper model (medium for good accuracy/speed balance)
    whisper_model = whisper.load_model("medium")
    
    # Load pyannote speaker diarization pipeline
    # Note: You need to accept pyannote/speaker-diarization-3.1 user agreement on HuggingFace
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True  # Set your HuggingFace token as environment variable
        )
        
        # Use GPU if available
        if torch.cuda.is_available():
            diarization_pipeline.to(torch.device("cuda"))
            logger.info("Using GPU for diarization")
        else:
            logger.info("Using CPU for diarization")
            
    except Exception as e:
        logger.error(f"Error loading diarization pipeline: {e}")
        logger.info("Please ensure you have accepted the pyannote/speaker-diarization-3.1 user agreement on HuggingFace")
        raise
    
    logger.info("Models loaded successfully")
    yield
    
    # Cleanup
    logger.info("Shutting down...")

app = FastAPI(
    title="Speaker Diarization API",
    description="Industry-leading speaker diarization API for South African call recordings",
    version="1.0.0",
    lifespan=lifespan
)

class AudioProcessor:
    """Handles audio preprocessing and format conversion"""
    
    @staticmethod
    def convert_to_wav(audio_data: bytes, format: str = "mp3") -> bytes:
        """Convert audio to WAV format"""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate to 16kHz (optimal for speech)
            audio = audio.set_frame_rate(16000)
            
            # Export as WAV
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")
    
    @staticmethod
    def reduce_noise(audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction to audio"""
        try:
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=audio_data, sr=sr)
            return reduced_noise
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data  # Return original if noise reduction fails

class SpeakerDiarizer:
    """Handles speaker diarization and transcription"""
    
    def __init__(self, whisper_model, diarization_pipeline):
        self.whisper_model = whisper_model
        self.diarization_pipeline = diarization_pipeline
    
    async def process_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Process audio file for speaker diarization and transcription"""
        try:
            # Load audio for diarization
            logger.info("Starting speaker diarization...")
            diarization = self.diarization_pipeline(audio_file_path)
            
            # Load audio for transcription
            logger.info("Loading audio for transcription...")
            audio_data, sr = librosa.load(audio_file_path, sr=16000)
            
            # Apply noise reduction
            audio_data = AudioProcessor.reduce_noise(audio_data, sr)
            
            # Get speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
            
            logger.info(f"Found {len(speaker_segments)} speaker segments")
            
            # Transcribe each segment
            transcribed_segments = []
            for segment in speaker_segments:
                # Extract audio segment
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                segment_audio = audio_data[start_sample:end_sample]
                
                # Transcribe segment
                if len(segment_audio) > 0:
                    result = self.whisper_model.transcribe(
                        segment_audio,
                        language='en',  # Specify English for South African context
                        task='transcribe'
                    )
                    
                    text = result['text'].strip()
                    if text:  # Only add non-empty transcriptions
                        transcribed_segments.append({
                            'start_time': segment['start'],
                            'end_time': segment['end'],
                            'speaker': segment['speaker'],
                            'text': text,
                            'confidence': self._calculate_confidence(result)
                        })
            
            # Format the final transcript
            formatted_transcript = self._format_transcript(transcribed_segments)
            
            return {
                'segments': transcribed_segments,
                'formatted_transcript': formatted_transcript,
                'total_duration': max([seg['end_time'] for seg in transcribed_segments]) if transcribed_segments else 0,
                'speaker_count': len(set([seg['speaker'] for seg in transcribed_segments]))
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate average confidence from Whisper result"""
        if 'segments' in whisper_result:
            confidences = []
            for segment in whisper_result['segments']:
                if 'avg_logprob' in segment:
                    # Convert log probability to confidence score
                    confidence = np.exp(segment['avg_logprob'])
                    confidences.append(confidence)
            return np.mean(confidences) if confidences else 0.8
        return 0.8  # Default confidence
    
    def _format_transcript(self, segments: List[Dict]) -> str:
        """Format transcript with proper speaker labels and timestamps"""
        if not segments:
            return ""
        
        formatted_lines = []
        formatted_lines.append("=== CALL TRANSCRIPT ===\n")
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        
        for segment in sorted_segments:
            # Format timestamp
            start_time = self._format_time(segment['start_time'])
            end_time = self._format_time(segment['end_time'])
            
            # Map speaker labels to more readable names
            speaker_name = self._map_speaker_name(segment['speaker'])
            
            formatted_lines.append(
                f"[{start_time} - {end_time}] {speaker_name}: {segment['text']}"
            )
        
        return "\n".join(formatted_lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _map_speaker_name(self, speaker_id: str) -> str:
        """Map speaker IDs to more readable names"""
        # For insurance context, assume first speaker is consultant, second is client
        speaker_mapping = {
            'SPEAKER_00': 'Consultant',
            'SPEAKER_01': 'Client',
            'SPEAKER_02': 'Speaker 3',
            'SPEAKER_03': 'Speaker 4'
        }
        return speaker_mapping.get(speaker_id, speaker_id)

# Initialize processor
audio_processor = AudioProcessor()

@app.post("/diarize")
async def diarize_audio(
    file: UploadFile = File(..., description="Audio file (MP3 or WAV)")
):
    """
    Process audio file for speaker diarization and transcription
    
    Returns:
    - Speaker segments with timestamps
    - Formatted transcript
    - Speaker count and duration
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_extension = file.filename.lower().split('.')[-1]
    if file_extension not in ['mp3', 'wav']:
        raise HTTPException(status_code=400, detail="Only MP3 and WAV files are supported")
    
    # Check file size (limit to 100MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 100MB")
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    
    # Generate unique request ID for logging
    request_id = str(uuid.uuid4())
    logger.info(f"Processing request {request_id} - File: {file.filename}, Size: {file_size} bytes")
    
    try:
        # Convert to WAV if needed
        if file_extension == 'mp3':
            wav_data = audio_processor.convert_to_wav(content, 'mp3')
        else:
            wav_data = content
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(wav_data)
            temp_file_path = temp_file.name
        
        try:
            # Process audio
            diarizer = SpeakerDiarizer(whisper_model, diarization_pipeline)
            result = await diarizer.process_audio(temp_file_path)
            
            # Add metadata
            result['request_id'] = request_id
            result['filename'] = file.filename
            result['processed_at'] = datetime.now().isoformat()
            
            logger.info(f"Request {request_id} processed successfully")
            return JSONResponse(content=result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": whisper_model is not None and diarization_pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Speaker Diarization API",
        "version": "1.0.0",
        "endpoints": {
            "diarize": "/diarize",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
