# Copyright (c) 2026 under the MIT License.
# SPDX-License-Identifier: MIT

import librosa
import numpy as np
from typing import Tuple, Dict, Any
from src.evaluation.metrics.base_audio_processor import BaseAudioProcessor
from src.evaluation.models import AudioMetrics


class AudioFeatureExtractor(BaseAudioProcessor):
    """
    Extracts comprehensive audio features for voice quality evaluation.

    This class implements the BaseAudioProcessor interface to provide
    standardized audio feature extraction using librosa. Features are
    categorized into fluency, naturalness, tone, and overall quality metrics.

    Attributes:
        sample_rate (int): Target sample rate for audio processing
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio feature extractor.

        Args:
            sample_rate: Target sample rate for audio resampling
                (default: 16000)
        """
        self.sample_rate = sample_rate

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and resample audio file to target sample rate.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            librosa.LibrosaError: If audio file is corrupted or unsupported
        """
        return librosa.load(audio_path, sr=self.sample_rate)

    def extract_features(
            self,
            audio_data: np.ndarray,
            sample_rate: int) -> AudioMetrics:
        """
        Extract comprehensive audio features from audio data.

        Orchestrates the extraction of all feature categories and returns
        a validated AudioMetrics object containing fluency, naturalness,
        tone, and overall quality metrics.

        Args:
            audio_data: Raw audio signal as numpy array
            sample_rate: Sample rate of the audio data

        Returns:
            AudioMetrics object containing all extracted features

        Raises:
            ValueError: If audio_data is empty or invalid
        """
        if len(audio_data) == 0:
            raise ValueError("Audio data cannot be empty")

        duration = len(audio_data) / sample_rate

        # Extract features by category (SRP compliance)
        fluency_metrics = self._extract_fluency_features(
            audio_data, sample_rate
        )
        naturalness_metrics = self._extract_naturalness_features(
            audio_data, sample_rate
        )
        tone_metrics = self._extract_tone_features(
            audio_data, sample_rate
        )
        quality_metrics = self._extract_quality_features(
            audio_data, sample_rate
        )

        return AudioMetrics(
            duration=duration,
            **fluency_metrics,
            **naturalness_metrics,
            **tone_metrics,
            **quality_metrics
        )

    def _extract_fluency_features(
            self,
            y: np.ndarray,
            sr: int) -> Dict[str, Any]:
        """
        Extract speech fluency and rhythm metrics.

        Analyzes pitch stability, speech rate, tempo, and energy consistency
        to measure speech fluency characteristics.

        Args:
            y: Audio signal array
            sr: Sample rate

        Returns:
            Dictionary containing fluency metrics
        """
        # Pitch analysis using YIN algorithm
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_valid = f0[f0 > 0]
        pitch_mean = np.mean(f0_valid) if len(f0_valid) > 0 else 0
        pitch_std = np.std(f0_valid) if len(f0_valid) > 0 else 0
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0

        # Speech rate from onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        speech_rate = len(onsets) / (len(y) / sr) if len(onsets) > 0 else 0

        # Tempo estimation
        tempo_estimates = librosa.beat.tempo(y=y, sr=sr)
        primary_tempo = tempo_estimates[0] if len(tempo_estimates) > 0 else 0

        # Energy consistency
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_consistency = (
            1 - (energy_std / energy_mean) if energy_mean > 0 else 0
        )

        return {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_cv': pitch_cv,
            'speech_rate': speech_rate,
            'primary_tempo': primary_tempo,
            'energy_mean': energy_mean,
            'energy_consistency': energy_consistency
        }

    def _extract_naturalness_features(
            self,
            y: np.ndarray,
            sr: int) -> Dict[str, Any]:
        """
        Extract naturalness and spectral characteristics.

        Analyzes spectral properties and MFCC coefficients to measure
        how natural the speech sounds.

        Args:
            y: Audio signal array
            sr: Sample rate

        Returns:
            Dictionary containing naturalness metrics
        """
        # Spectral characteristics
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr)
        )
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=y, sr=sr)
        )

        # MFCC coefficients (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()

        return {
            'spectral_centroid': spectral_centroid,
            'mfcc_coefficients': mfcc_mean,
            'spectral_rolloff': spectral_rolloff
        }

    def _extract_tone_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract tonal and harmonic characteristics.

        Analyzes spectral contrast, zero crossing rate, and harmonic-to-noise
        ratio to measure tonal quality of speech.

        Args:
            y: Audio signal array
            sr: Sample rate

        Returns:
            Dictionary containing tone metrics
        """
        # Spectral contrast and zero crossing rate
        spectral_contrast = np.mean(
            librosa.feature.spectral_contrast(y=y, sr=sr)
        )
        zero_crossing_rate = np.mean(
            librosa.feature.zero_crossing_rate(y)
        )

        # Harmonic-to-noise ratio calculation
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        harmonic_energy = np.sum(magnitude**2)
        noise_energy = np.sum(
            (magnitude - np.mean(magnitude, axis=1, keepdims=True))**2)
        hnr = (
            10 * np.log10(harmonic_energy / (noise_energy + 1e-10))
            if noise_energy > 0 else 0
        )

        return {
            'spectral_contrast': spectral_contrast,
            'zero_crossing_rate': zero_crossing_rate,
            'harmonic_noise_ratio': hnr
        }

    def _extract_quality_features(
            self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract overall audio quality metrics.

        Analyzes chroma, tonnetz, dynamic range, and spectral flatness
        to measure overall audio quality characteristics.

        Args:
            y: Audio signal array
            sr: Sample rate

        Returns:
            Dictionary containing quality metrics
        """
        # Harmonic and tonal features
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        tonnetz = np.mean(
            librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        )

        # Dynamic range and spectral flatness
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range = np.max(rms) / (np.min(rms) + 1e-10)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        return {
            'chroma': chroma,
            'tonnetz': tonnetz,
            'dynamic_range': dynamic_range,
            'spectral_flatness': spectral_flatness
        }
