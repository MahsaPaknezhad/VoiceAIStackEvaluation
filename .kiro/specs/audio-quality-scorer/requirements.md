# Requirements Document

## Introduction

A standalone Python script that accepts a `.wav` audio file as input and uses the MiniCPM-o multimodal model to evaluate the audio, producing a naturalness score and a clarity score. The script lives in `server/scripts/` and integrates with the existing evaluation infrastructure in the project.

## Glossary

- **Script**: The standalone Python script `score_audio.py` that performs audio quality evaluation
- **MiniCPM-o**: The OpenBMB multimodal model used for audio understanding and scoring (https://github.com/OpenBMB/MiniCPM-o)
- **Naturalness_Score**: A numeric score (0–10) representing how natural and human-like the speech in the audio sounds
- **Clarity_Score**: A numeric score (0–10) representing how clear and intelligible the speech in the audio is
- **WAV_File**: A `.wav` format audio file provided as input to the Script
- **Model**: The loaded MiniCPM-o model instance used for inference

## Requirements

### Requirement 1: Accept WAV Audio File Input

**User Story:** As a developer, I want to pass a `.wav` file path as a command-line argument, so that I can evaluate any audio file without modifying the script.

#### Acceptance Criteria

1. THE Script SHALL accept a positional command-line argument specifying the path to a WAV_File
2. WHEN the provided file path does not exist, THE Script SHALL print a descriptive error message and exit with a non-zero exit code
3. WHEN the provided file path does not have a `.wav` extension, THE Script SHALL print a descriptive error message and exit with a non-zero exit code
4. WHEN a valid WAV_File path is provided, THE Script SHALL proceed to load and evaluate the audio

### Requirement 2: Load and Initialize MiniCPM-o Model

**User Story:** As a developer, I want the script to load the MiniCPM-o model automatically, so that I don't need to manage model initialization separately.

#### Acceptance Criteria

1. THE Script SHALL load the MiniCPM-o model from the `openbmb/MiniCPM-o-2_6` Hugging Face checkpoint
2. WHEN the Model fails to load due to missing dependencies or insufficient resources, THE Script SHALL print a descriptive error message and exit with a non-zero exit code
3. WHERE a CUDA-capable GPU is available, THE Script SHALL load the Model onto the GPU device
4. WHERE no CUDA-capable GPU is available, THE Script SHALL load the Model onto the CPU device

### Requirement 3: Evaluate Audio Naturalness

**User Story:** As a developer, I want the script to produce a naturalness score for the audio, so that I can quantify how human-like the speech sounds.

#### Acceptance Criteria

1. WHEN a valid WAV_File is loaded, THE Script SHALL prompt the Model to evaluate the naturalness of the speech in the audio
2. THE Script SHALL extract a Naturalness_Score in the range 0 to 10 from the Model's response
3. WHEN the Model response does not contain a parseable numeric Naturalness_Score, THE Script SHALL print a descriptive error message and exit with a non-zero exit code

### Requirement 4: Evaluate Audio Clarity

**User Story:** As a developer, I want the script to produce a clarity score for the audio, so that I can quantify how intelligible the speech is.

#### Acceptance Criteria

1. WHEN a valid WAV_File is loaded, THE Script SHALL prompt the Model to evaluate the clarity of the speech in the audio
2. THE Script SHALL extract a Clarity_Score in the range 0 to 10 from the Model's response
3. WHEN the Model response does not contain a parseable numeric Clarity_Score, THE Script SHALL print a descriptive error message and exit with a non-zero exit code

### Requirement 5: Output Scores

**User Story:** As a developer, I want the scores printed to stdout in a consistent format, so that I can parse the output programmatically or read it at a glance.

#### Acceptance Criteria

1. WHEN evaluation completes successfully, THE Script SHALL print the Naturalness_Score and Clarity_Score to stdout
2. THE Script SHALL output scores in the format: `naturalness: <score>\nclarity: <score>` where `<score>` is a float rounded to two decimal places
3. THE Script SHALL exit with exit code 0 upon successful evaluation

### Requirement 6: Single Combined Evaluation Prompt

**User Story:** As a developer, I want both scores produced in a single model call where possible, so that inference time is minimised.

#### Acceptance Criteria

1. THE Script SHALL request both the Naturalness_Score and the Clarity_Score from the Model in a single inference call
2. WHEN the Model returns both scores in a single response, THE Script SHALL parse both values from that response
3. IF the Model response contains only one of the two scores, THEN THE Script SHALL print a descriptive error message indicating which score is missing and exit with a non-zero exit code
