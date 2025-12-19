#!/usr/bin/env python3
"""
Quick test to verify AWS Polly PCM audio output.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).parent))

import boto3
from contextlib import closing

# Get credentials from environment
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', '')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

print("🔧 Testing AWS Polly with PCM format...")

polly = boto3.client(
    'polly',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Test text
test_text = "Hello! This is a test of AWS Polly with PCM audio format."

print(f"🔊 Generating speech: '{test_text}'")

# Generate speech with PCM format
response = polly.synthesize_speech(
    Text=test_text,
    OutputFormat='pcm',
    VoiceId='Salli',
    SampleRate='24000'
)

# Save to file
output_file = 'test_pcm_output.pcm'
with closing(response['AudioStream']) as stream:
    with open(output_file, 'wb') as file:
        data = stream.read()
        file.write(data)
        print(f"✅ Generated {len(data)} bytes of PCM audio")
        print(f"📁 Saved to: {output_file}")

print("\n✅ PCM format test successful!")
print("The audio should now work properly in your voice interface.")
