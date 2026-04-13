# tests/test_pipeline_integration.py

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.voice_pipeline_evaluator import VoiceAssistantRunner


class TestPipelineIntegration:
    """Integration tests for the voice pipeline"""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing"""
        return {
            "questions": [
                {
                    "id": "test_001",
                    "text": "What is the weather today?",
                    "audio_file": "test_001.wav"
                }
            ]
        }
    
    @pytest.fixture
    def temp_dataset_file(self, mock_dataset):
        """Create temporary dataset file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mock_dataset, f)
            return f.name
    
    def test_timing_collector_initialization(self, temp_dataset_file):
        """Test that TimingCollector is properly initialized"""
        runner = VoiceAssistantRunner(
            dataset_path=temp_dataset_file,
            audio_dir="/tmp"
        )
        
        # Verify runner initializes correctly
        assert runner.dataset_path == temp_dataset_file
        assert runner.audio_dir == "/tmp"
        assert len(runner.dataset['questions']) == 1
    
    def teardown_method(self):
        """Clean up temporary files"""
        # Clean up any temp files created during tests
        pass
