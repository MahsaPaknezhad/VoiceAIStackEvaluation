
#!/usr/bin/env python3
"""
Test script to verify LiveKit NVIDIA TTS connection to self-hosted Riva server
This tests the connection using LiveKit's native NVIDIA integration
"""

import grpc
import sys

# Configuration
RIVA_SERVER = "[IP ADDRESS]"

def test_basic_grpc_connection():
    """Test basic insecure gRPC connection to Riva server"""
    print(f"Testing basic gRPC connection to {RIVA_SERVER}...")
    
    try:
        channel = grpc.insecure_channel(RIVA_SERVER)
        grpc.channel_ready_future(channel).result(timeout=10)
        print("✅ SUCCESS: Basic gRPC connection established!")
        channel.close()
        return True
    except grpc.FutureTimeoutError:
        print("❌ FAILED: Connection timeout")
        return False
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)}")
        return False

def test_livekit_nvidia_tts():
    """Test LiveKit's NVIDIA TTS integration with self-hosted Riva"""
    print(f"Testing LiveKit NVIDIA TTS with self-hosted Riva...")
    
    try:
        # Try importing LiveKit's NVIDIA plugin
        try:
            from livekit.plugins import nvidia
            print("✅ LiveKit NVIDIA plugin found")
        except ImportError:
            print("⚠️  LiveKit NVIDIA plugin not found")
            print("   Try: pip install livekit-plugins-nvidia")
            return False
        
        # Try different configuration approaches
        configs = [
            {
                'name': 'Standard config with use_ssl=False',
                'params': {
                    'server': RIVA_SERVER,
                    'voice': 'Magpie-Multilingual.EN-US.Aria',
                    'use_ssl': False
                }
            },
            {
                'name': 'Config with insecure=True',
                'params': {
                    'server': RIVA_SERVER,
                    'voice': 'Magpie-Multilingual.EN-US.Aria',
                    'insecure': True
                }
            },
            {
                'name': 'Config with secure=False',
                'params': {
                    'server': RIVA_SERVER,
                    'voice': 'Magpie-Multilingual.EN-US.Aria',
                    'secure': False
                }
            },
            {
                'name': 'Config with grpc:// scheme',
                'params': {
                    'server': f'grpc://{RIVA_SERVER}',
                    'voice': 'Magpie-Multilingual.EN-US.Aria'
                }
            },
        ]
        
        for config in configs:
            try:
                print(f"Testing: {config['name']}")
                tts = nvidia.TTS(**config['params'])
                print(f"  ✅ SUCCESS: {config['name']} works!")
                print(f"     Configuration: {config['params']}")
                return True
            except TypeError as e:
                print(f"  ❌ Invalid parameters: {e}")
            except Exception as e:
                print(f"  ❌ Failed: {type(e).__name__}: {str(e)}")
        
        print("❌ All LiveKit configurations failed")
        return False
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)}")
        return False

def test_livekit_alternative_imports():
    """Test alternative LiveKit import paths for NVIDIA support"""
    print(f"Testing alternative LiveKit import paths...")
    
    import_paths = [
        'livekit.plugins.nvidia',
        'livekit.agents.tts.nvidia',
        'livekit_plugins_nvidia',
        'livekit.tts.nvidia',
    ]
    
    for path in import_paths:
        try:
            parts = path.split('.')
            module = __import__(path)
            for part in parts[1:]:
                module = getattr(module, part)
            print(f"  ✅ Found: {path}")
            print(f"     Available classes: {[x for x in dir(module) if not x.startswith('_')]}")
            return True
        except ImportError:
            print(f"  ❌ Not found: {path}")
        except Exception as e:
            print(f"  ⚠️  Error with {path}: {e}")
    
    return False

def test_riva_client_direct():
    """Test direct Riva client connection (fallback option)"""
    print(f"Testing direct Riva client connection...")
    
    try:
        import riva.client
        
        # Create auth WITHOUT SSL
        auth = riva.client.Auth(
            ssl_cert=None,
            use_ssl=False,
            server=RIVA_SERVER,
            metadata=None
        )
        
        service = riva.client.SpeechSynthesisService(auth)
        print("✅ SUCCESS: Direct Riva client connection established!")
        print("   You can use riva.client directly if LiveKit doesn't support self-hosted")
        return True
        
    except ImportError:
        print("⚠️  riva.client not found")
        print("   Install with: pip install nvidia-riva-client")
        return False
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("LiveKit NVIDIA TTS Connection Test")
    print("=" * 70)
    
    # Test 1: Basic gRPC connection
    test1 = test_basic_grpc_connection()
    
    # Test 2: LiveKit NVIDIA TTS
    test2 = test_livekit_nvidia_tts()
    
    # Test 3: Alternative import paths
    test3 = test_livekit_alternative_imports()
    
    # Test 4: Direct Riva client (fallback)
    test4 = test_riva_client_direct()
    
    print("" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Basic gRPC connection: {'✅ WORKS' if test1 else '❌ FAILED'}")
    print(f"LiveKit NVIDIA TTS: {'✅ WORKS' if test2 else '❌ FAILED'}")
    print(f"Alternative imports: {'✅ FOUND' if test3 else '❌ NOT FOUND'}")
    print(f"Direct Riva client: {'✅ WORKS' if test4 else '❌ FAILED'}")
    
    if test2:
        print("✅ LiveKit native NVIDIA support works!")
        print("   Use the successful configuration in your voice agent")
    elif test4:
        print("⚠️  LiveKit doesn't support self-hosted Riva")
        print("   Use direct riva.client integration instead")
        print("   Or create a custom LiveKit TTS plugin")
    else:
        print("❌ No working configuration found")
        print("   Recommendations:")
        print("   1. Check LiveKit documentation for NVIDIA Riva support")
        print("   2. Install: pip install livekit-plugins-nvidia")
        print("   3. Use direct riva.client as fallback")

