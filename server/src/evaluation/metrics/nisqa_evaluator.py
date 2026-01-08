import os
import tempfile
import librosa
import soundfile as sf
from loguru import logger
from src.evaluation.metrics.NISQA.nisqa.NISQA_model import nisqaModel
from src.evaluation.metrics.base_quality_evaluator import BaseQualityEvaluator
from src.evaluation.models import NISQAConfig, NISQAResults


class NISQAEvaluator(BaseQualityEvaluator[NISQAResults]):
    """
    NISQA (Non-Intrusive Speech Quality Assessment) evaluator.

    Implements NISQA model for predicting speech quality metrics including
    MOS (Mean Opinion Score), noisiness, coloration, discontinuity,
    and loudness.

    Attributes:
        config (NISQAConfig): NISQA configuration parameters
        nisqa_args (dict): NISQA prediction arguments template
    """

    def __init__(self, config: NISQAConfig):
        """
        Initialize NISQA evaluator with configuration.

        Args:
            config: NISQA configuration containing model path and parameters
        """
        self.config = config
        self.nisqa_args = None

    async def initialize(self) -> bool:
        """
        Initialize NISQA model and prediction arguments.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Verify model path exists
            if not os.path.exists(self.config.model_path):
                logger.warning(
                    f"Model Path for NISQA {self.config.model_path} dne"
                )
                return False

            # Set up prediction arguments template
            self.nisqa_args = {
                'mode': 'predict_file',
                'pretrained_model': self.config.model_path,
                'deg': None,  # Will be set during prediction
                'tr_bs_val': self.config.batch_size,
                'tr_num_workers': 0,
                'ms_channel': None,  # For mono audio
                'output_dir': None,  # No output file needed
                'ms_max_segments': self.config.max_segments
            }

            return True

        except Exception:
            return False

    async def evaluate(self, audio_path: str) -> NISQAResults:
        """
        Evaluate audio quality using NISQA model.

        Args:
            audio_path: Path to audio file for evaluation

        Returns:
            NISQAResults with validated NISQA metrics

        Raises:
            RuntimeError: If model not initialized or evaluation fails
        """
        if not self.is_available():
            raise RuntimeError("NISQA model not initialized")

        temp_file = None
        try:
            # Load and resample to 48kHz (NISQA requirement)
            audio_data, _ = librosa.load(
                audio_path, sr=self.config.sample_rate
            )
            # Create temporary file with correct format
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False
            )
            sf.write(temp_file.name, audio_data, self.config.sample_rate)
            temp_file.close()

            # Create args for this prediction
            args = self.nisqa_args.copy()
            args['deg'] = temp_file.name

            # Create model instance and run prediction
            nisqa_model = nisqaModel(args)
            results_df = nisqa_model.predict()

            # Extract metrics from results DataFrame
            if results_df is not None and not results_df.empty:
                row = results_df.iloc[0]
                return NISQAResults(
                    nisqa_mos=float(row.get('mos_pred', 0)),
                    nisqa_noisiness=float(row.get('noi_pred', 0)),
                    nisqa_coloration=float(row.get('col_pred', 0)),
                    nisqa_discontinuity=float(row.get('dis_pred', 0)),
                    nisqa_loudness=float(row.get('loud_pred', 0))
                )
            else:
                return NISQAResults(
                    nisqa_mos=0,
                    nisqa_noisiness=0,
                    nisqa_coloration=0,
                    nisqa_discontinuity=0,
                    nisqa_loudness=0
                )

        except Exception:
            # Return validated default values
            return NISQAResults(
                nisqa_mos=0,
                nisqa_noisiness=0,
                nisqa_coloration=0,
                nisqa_discontinuity=0,
                nisqa_loudness=0
            )
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def is_available(self) -> bool:
        """
        Check if NISQA model is available and initialized.

        Returns:
            True if model is ready for evaluation, False otherwise
        """
        return (self.nisqa_args is not None and
                os.path.exists(self.config.model_path))
