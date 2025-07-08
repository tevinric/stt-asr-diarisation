#!/usr/bin/env python3
"""
Test script to verify Speaker Diarization API setup
"""

import os
import sys
import requests
import tempfile
import time
from pathlib import Path
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

def create_test_audio():
    """Create a simple test audio file with two distinct speakers"""
    print("Creating test audio file...")
    
    # Create two different tones to simulate different speakers
    # Speaker 1: 440Hz tone (A note)
    speaker1 = Sine(440).to_audio_segment(duration=3000)  # 3 seconds
    
    # Brief silence
    silence = AudioSegment.silent(duration=500)  # 0.5 seconds
    
    # Speaker 2: 880Hz tone (A note, one octave higher)
    speaker2 = Sine(880).to_audio_segment(duration=3000)  # 3 seconds
    
    # Another brief silence
    silence2 = AudioSegment.silent(duration=500)
    
    # Speaker 1 again
    speaker1_again = Sine(440).to_audio_segment(duration=2000)  # 2 seconds
    
    # Combine all segments
    test_audio = speaker1 + silence + speaker2 + silence2 + speaker1_again
    
    # Export as MP3
    test_file = "test_audio.mp3"
    test_audio.export(test_file, format="mp3")
    
    print(f"✓ Created test audio file: {test_file} ({len(test_audio)/1000:.1f}s)")
    return test_file

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'whisper',
        'pyannote.audio',
        'torch',
        'pydub',
        'noisereduce',
        'librosa',
        'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_huggingface_auth():
    """Check HuggingFace authentication"""
    print("\nChecking HuggingFace authentication...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"✗ HuggingFace authentication failed: {e}")
        print("Please run: huggingface-cli login")
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    
    try:
        # Test Whisper
        print("Loading Whisper model...")
        import whisper
        model = whisper.load_model("base")  # Use smaller model for testing
        print("✓ Whisper model loaded")
        
        # Test pyannote
        print("Loading pyannote model...")
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        print("✓ pyannote model loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_api_server():
    """Test if the API server is running"""
    print("\nTesting API server...")
    
    base_url = "http://localhost:8000"
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✓ API server is running")
                print(f"  Status: {health_data.get('status')}")
                print(f"  Models loaded: {health_data.get('models_loaded')}")
                return True
            else:
                print(f"✗ API returned status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                print("API server not running. Attempting to start...")
                print("Please run in another terminal: python main.py")
                print("Waiting for server to start...")
            
            time.sleep(5)
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            
        except Exception as e:
            print(f"✗ Error connecting to API: {e}")
            break
    
    print("✗ Could not connect to API server")
    print("Please ensure the server is running: python main.py")
    return False

def test_diarization():
    """Test the actual diarization functionality"""
    print("\nTesting speaker diarization...")
    
    # Create test audio
    test_file = create_test_audio()
    
    try:
        # Test API endpoint
        url = "http://localhost:8000/diarize"
        
        with open(test_file, 'rb') as f:
            files = {'file': f}
            print("Sending test file to API...")
            response = requests.post(url, files=files, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Diarization successful!")
            print(f"  Filename: {result.get('filename')}")
            print(f"  Duration: {result.get('total_duration', 0):.1f}s")
            print(f"  Speakers found: {result.get('speaker_count', 0)}")
            print(f"  Segments: {len(result.get('segments', []))}")
            print(f"  Processing time: {result.get('processing_time', 0):.1f}s")
            
            # Show sample transcript
            if result.get('formatted_transcript'):
                print("\nSample transcript:")
                print("-" * 40)
                print(result['formatted_transcript'][:300] + "..." if len(result['formatted_transcript']) > 300 else result['formatted_transcript'])
            
            return True
        else:
            print(f"✗ API request failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Diarization test failed: {e}")
        return False
    
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up test file: {test_file}")

def main():
    """Run all tests"""
    print("🎤 Speaker Diarization API Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Check dependencies
    if not check_dependencies():
        all_tests_passed = False
    
    # Check HuggingFace auth
    if not check_huggingface_auth():
        all_tests_passed = False
    
    # Test model loading (optional - can be slow)
    print(f"\nWould you like to test model loading? This may take several minutes. (y/n): ", end="")
    if input().lower().startswith('y'):
        if not test_model_loading():
            all_tests_passed = False
    
    # Test API server
    if not test_api_server():
        all_tests_passed = False
        print("\n⚠️  Cannot test diarization without running API server")
    else:
        # Test actual diarization
        if not test_diarization():
            all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your Speaker Diarization API is ready to use!")
        print("\nNext steps:")
        print("1. Start the API server: python main.py")
        print("2. Use the client: python client_example.py")
        print("3. Send audio files to: http://localhost:8000/diarize")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above and fix them before proceeding.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Set up HuggingFace: huggingface-cli login")
        print("- Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("- Start API server: python main.py")

if __name__ == "__main__":
    main()
