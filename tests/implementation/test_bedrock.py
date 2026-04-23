# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

# test_bedrock.py
import boto3
import os
import json
from dotenv import load_dotenv
from botocore.config import Config

load_dotenv()

def test_bedrock():
    print("Testing Bedrock connection...")
    
    try:
        # Create Bedrock client with timeout
        config = Config(
            read_timeout=10,
            connect_timeout=10,
            retries={'max_attempts': 1}
        )
        
        bedrock = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            config=config
        )
        
        print("Client created, testing ConverseStream with Strands model...")
        
        # Test the streaming API with the Strands model
        response = bedrock.converse_stream(
            modelId='us.anthropic.claude-haiku-4-5-20251001-v1:0',
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            inferenceConfig={"maxTokens": 10}
        )
        
        print("✓ Bedrock ConverseStream with Strands model is working!")
        return True
        
    except Exception as e:
        print(f"✗ Bedrock error: {e}")
        return False

if __name__ == "__main__":
    test_bedrock()
