import requests
import json
import time
import os
from typing import List, Dict, Any
from pathlib import Path
import concurrent.futures
from datetime import datetime

class SpeakerDiarizationClient:
    """Client for the Speaker Diarization API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy and models are loaded"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def diarize_file(self, file_path: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Process a single audio file for speaker diarization
        
        Args:
            file_path: Path to the audio file
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary containing diarization results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise ValueError(f"File too large: {file_size} bytes. Maximum is 100MB")
        
        print(f"Processing: {file_path} ({file_size:,} bytes)")
        start_time = time.time()
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                response = self.session.post(
                    f"{self.base_url}/diarize",
                    files=files,
                    timeout=timeout
                )
                response.raise_for_status()
                
            processing_time = time.time() - start_time
            result = response.json()
            result['processing_time'] = processing_time
            
            print(f"✓ Processed in {processing_time:.1f}s - Found {result['speaker_count']} speakers")
            return result
            
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def batch_process(self, file_paths: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        Process multiple files concurrently
        
        Args:
            file_paths: List of audio file paths
            max_workers: Maximum number of concurrent requests
            
        Returns:
            List of results for each file
        """
        print(f"Starting batch processing of {len(file_paths)} files...")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.diarize_file, file_path): file_path 
                for file_path in file_paths
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    result['source_file'] = file_path
                    results.append(result)
                except Exception as e:
                    print(f"✗ Failed to process {file_path}: {e}")
                    results.append({
                        'source_file': file_path,
                        'error': str(e),
                        'success': False
                    })
        
        successful = sum(1 for r in results if 'error' not in r)
        print(f"Batch complete: {successful}/{len(file_paths)} files processed successfully")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = "output"):
        """Save diarization results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for result in results:
            if 'error' in result:
                continue
                
            filename = result.get('filename', 'unknown')
            base_name = os.path.splitext(filename)[0]
            
            # Save full JSON result
            json_path = os.path.join(output_dir, f"{base_name}_result.json")
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Save formatted transcript
            if 'formatted_transcript' in result:
                txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
                with open(txt_path, 'w') as f:
                    f.write(result['formatted_transcript'])
            
            print(f"✓ Saved results for {filename}")

def main():
    """Example usage of the Speaker Diarization API"""
    
    # Initialize client
    client = SpeakerDiarizationClient()
    
    # Check API health
    print("Checking API health...")
    health = client.health_check()
    print(f"API Status: {health.get('status')}")
    
    if health.get('status') != 'healthy':
        print("⚠️  API is not healthy. Please check the server.")
        return
    
    # Example 1: Process a single file
    print("\n=== Single File Processing ===")
    try:
        # Replace with your audio file path
        audio_file = "sample_call.mp3"
        
        if os.path.exists(audio_file):
            result = client.diarize_file(audio_file)
            
            print(f"\nResults for {audio_file}:")
            print(f"Duration: {result['total_duration']:.1f}s")
            print(f"Speakers: {result['speaker_count']}")
            print(f"Segments: {len(result['segments'])}")
            
            print("\nFormatted Transcript:")
            print("-" * 50)
            print(result['formatted_transcript'])
            
        else:
            print(f"Sample file not found: {audio_file}")
    
    except Exception as e:
        print(f"Error processing single file: {e}")
    
    # Example 2: Batch processing
    print("\n=== Batch Processing ===")
    
    # Find all audio files in a directory
    audio_dir = "audio_files"  # Replace with your directory
    if os.path.exists(audio_dir):
        audio_files = []
        for ext in ['*.mp3', '*.wav']:
            audio_files.extend(Path(audio_dir).glob(ext))
        
        if audio_files:
            file_paths = [str(f) for f in audio_files[:5]]  # Limit to 5 files for demo
            
            try:
                results = client.batch_process(file_paths, max_workers=2)
                
                # Save all results
                client.save_results(results)
                
                # Print summary
                print("\n=== Batch Results Summary ===")
                for result in results:
                    if 'error' not in result:
                        print(f"✓ {result['filename']}: {result['speaker_count']} speakers, "
                              f"{result['total_duration']:.1f}s duration")
                    else:
                        print(f"✗ {result['source_file']}: {result['error']}")
                        
            except Exception as e:
                print(f"Batch processing failed: {e}")
        else:
            print(f"No audio files found in {audio_dir}")
    else:
        print(f"Audio directory not found: {audio_dir}")

def analyze_results(results_file: str):
    """Analyze and extract insights from diarization results"""
    
    with open(results_file, 'r') as f:
        result = json.load(f)
    
    segments = result['segments']
    
    # Calculate speaking time per speaker
    speaker_times = {}
    for segment in segments:
        speaker = segment['speaker']
        duration = segment['end_time'] - segment['start_time']
        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
    
    # Map to readable names
    speaker_mapping = {
        'SPEAKER_00': 'Consultant',
        'SPEAKER_01': 'Client'
    }
    
    print(f"\n=== Call Analysis ===")
    print(f"Total Duration: {result['total_duration']:.1f}s")
    print(f"Total Segments: {len(segments)}")
    print("\nSpeaking Time Distribution:")
    
    for speaker_id, time in speaker_times.items():
        speaker_name = speaker_mapping.get(speaker_id, speaker_id)
        percentage = (time / result['total_duration']) * 100
        print(f"  {speaker_name}: {time:.1f}s ({percentage:.1f}%)")
    
    # Find longest segments
    longest_segments = sorted(segments, key=lambda x: x['end_time'] - x['start_time'], reverse=True)[:3]
    
    print("\nLongest Speaking Segments:")
    for i, segment in enumerate(longest_segments, 1):
        speaker_name = speaker_mapping.get(segment['speaker'], segment['speaker'])
        duration = segment['end_time'] - segment['start_time']
        print(f"  {i}. {speaker_name}: {duration:.1f}s - \"{segment['text'][:100]}...\"")

if __name__ == "__main__":
    main()
    
    # Example: Analyze a specific result file
    # analyze_results("output/sample_call_result.json")
