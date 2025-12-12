#!/usr/bin/env python3
"""
Test AWS credentials for Pipecat AWS Transcribe service
"""
import os
import boto3
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

def test_boto3_credentials():
    """Test if boto3 can access AWS credentials"""
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            logger.info("✅ Boto3 credentials found")
            logger.info(f"Access Key: {credentials.access_key[:10]}...")
            logger.info(f"Has session token: {bool(credentials.token)}")
            
            # Test STS call
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS Identity: {identity['Arn']}")
            return True
        else:
            logger.error("❌ No boto3 credentials found")
            return False
    except Exception as e:
        logger.error(f"❌ Boto3 credential test failed: {e}")
        return False

def test_pipecat_aws_transcribe():
    """Test Pipecat AWS Transcribe service creation"""
    try:
        from pipecat.services.aws.stt import AWSTranscribeSTTService
        
        # Test with no explicit credentials (should use IAM role)
        logger.info("Testing Pipecat AWS Transcribe with default credentials...")
        
        stt = AWSTranscribeSTTService(
            region="ap-southeast-2",
            session_timeout=60000,
            enable_partial_results_stabilization=True,
            partial_results_stability="high"
        )
        
        logger.info("✅ Pipecat AWS Transcribe service created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipecat AWS Transcribe test failed: {e}")
        return False

def test_transcribe_client_direct():
    """Test direct AWS Transcribe client"""
    try:
        import boto3
        
        client = boto3.client('transcribe', region_name='ap-southeast-2')
        
        # Test a simple API call
        response = client.list_transcription_jobs(MaxResults=1)
        logger.info("✅ Direct AWS Transcribe client works")
        logger.info(f"Found {len(response.get('TranscriptionJobSummaries', []))} transcription jobs")
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct AWS Transcribe test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AWS Credentials Test for Pipecat")
    print("=" * 60)
    
    test1 = test_boto3_credentials()
    test2 = test_pipecat_aws_transcribe()
    test3 = test_transcribe_client_direct()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Boto3 credentials: {'✅ WORKS' if test1 else '❌ FAILED'}")
    print(f"Pipecat AWS Transcribe: {'✅ WORKS' if test2 else '❌ FAILED'}")
    print(f"Direct Transcribe client: {'✅ WORKS' if test3 else '❌ FAILED'}")
    
    if all([test1, test2, test3]):
        print("\n✅ All AWS credential tests passed!")
        print("The full pipeline should work now.")
    else:
        print("\n❌ Some tests failed. Check IAM role permissions.")
        print("Required permissions: transcribe:StartStreamTranscription")