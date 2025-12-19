#!/usr/bin/env python3
"""
Test script for AWS Polly TTS integration.
This script tests the AWS Polly TTS handler independently.
"""
import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_polly_basic():
    """Test basic AWS Polly functionality."""
    import boto3
    from contextlib import closing
    
    # Get credentials from environment
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', '')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not aws_access_key_id or not aws_secret_access_key:
        print("❌ AWS credentials not found in environment variables")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file")
        return False
    
    try:
        print("🔧 Initializing AWS Polly client...")
        polly = boto3.client(
            'polly',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        print(f"✅ AWS Polly client initialized (region: {aws_region})")
        
        # Test text
        test_text = "Hello! This is a test of AWS Polly text to speech integration."
        
        print(f"🔊 Generating speech for: '{test_text}'")
        
        # Generate speech
        response = polly.synthesize_speech(
            Text=test_text,
            OutputFormat='mp3',
            VoiceId='Salli'
        )
        
        # Save to file
        output_file = 'test_polly_output.mp3'
        with closing(response['AudioStream']) as stream:
            with open(output_file, 'wb') as file:
                file.write(stream.read())
        
        print(f"✅ Speech generated successfully!")
        print(f"📁 Audio saved to: {output_file}")
        print(f"🎵 You can play this file to verify the audio quality")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AWS Polly: {e}")
        return False


def test_polly_handler():
    """Test the AWSPollyTTSHandler class."""
    import asyncio
    
    # Set TTS provider to polly
    os.environ['TTS_PROVIDER'] = 'polly'
    
    try:
        # Import after setting environment
        from text_to_voice_server import AWSPollyTTSHandler
        
        print("\n🔧 Testing AWSPollyTTSHandler class...")
        
        handler = AWSPollyTTSHandler()
        print("✅ AWSPollyTTSHandler initialized successfully")
        
        # Test streaming
        async def test_stream():
            test_text = "Testing the AWS Polly TTS handler with streaming support."
            print(f"🔊 Testing streaming for: '{test_text}'")
            
            chunks = []
            async for chunk in handler.text_to_speech_stream(test_text):
                chunks.append(chunk)
            
            print(f"✅ Received {len(chunks)} audio chunks")
            
            # Save combined audio
            output_file = 'test_handler_output.mp3'
            with open(output_file, 'wb') as f:
                for chunk in chunks:
                    f.write(chunk)
            
            print(f"📁 Audio saved to: {output_file}")
            return True
        
        result = asyncio.run(test_stream())
        return result
        
    except Exception as e:
        print(f"❌ Error testing AWSPollyTTSHandler: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("AWS Polly TTS Integration Test")
    print("=" * 60)
    
    # Test 1: Basic Polly functionality
    print("\n📋 Test 1: Basic AWS Polly API")
    print("-" * 60)
    test1_result = test_polly_basic()
    
    # Test 2: Handler class
    print("\n📋 Test 2: AWSPollyTTSHandler Class")
    print("-" * 60)
    test2_result = test_polly_handler()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Basic AWS Polly API: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"AWSPollyTTSHandler:  {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! AWS Polly TTS is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Set TTS_PROVIDER=polly in your .env file")
        print("   2. Restart your socket server")
        print("   3. Test with the voice interface")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)
