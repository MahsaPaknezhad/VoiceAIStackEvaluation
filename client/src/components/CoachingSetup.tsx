import React, { useState } from 'react';
import TranscriptSelector from './TranscriptSelector';
import SelfAssessmentTable from './SelfAssessmentTable';
import { type Transcript } from '../data/transcripts';

interface SelfAssessmentData {
  wentWell: string;
  improvements: string;
}

interface CoachingSetupProps {
  onSetupComplete: (data: { transcript: Transcript; assessment: SelfAssessmentData }) => void;
}

const CoachingSetup: React.FC<CoachingSetupProps> = ({ onSetupComplete }) => {
  const [selectedTranscript, setSelectedTranscript] = useState<Transcript | null>(null);
  const [assessment, setAssessment] = useState<SelfAssessmentData | null>(null);
  const [currentStep, setCurrentStep] = useState<'transcript' | 'assessment' | 'complete'>('transcript');

  const handleTranscriptSelect = (transcript: Transcript | null) => {
    setSelectedTranscript(transcript);
    if (transcript) {
      setCurrentStep('assessment');
    }
  };

  const handleAssessmentComplete = (assessmentData: SelfAssessmentData) => {
    setAssessment(assessmentData);
    setCurrentStep('complete');
    if (selectedTranscript) {
      onSetupComplete({
        transcript: selectedTranscript,
        assessment: assessmentData
      });
    }
  };

  const handleStartOver = () => {
    setSelectedTranscript(null);
    setAssessment(null);
    setCurrentStep('transcript');
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-2xl font-bold text-foreground mb-2">
          🎯 NAB AI Coaching Setup
        </h1>
        <p className="text-muted-foreground">
          Select a transcript and complete your self-reflection to begin coaching
        </p>
      </div>

      {/* Step 1: Transcript Selection */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
            currentStep !== 'transcript' ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
          }`}>
            {selectedTranscript ? '✓' : '1'}
          </div>
          <h2 className="text-lg font-semibold">Select Transcript</h2>
        </div>

        <div className={currentStep !== 'transcript' ? 'opacity-60' : ''}>
          <TranscriptSelector
            onTranscriptSelect={handleTranscriptSelect}
            selectedTranscript={selectedTranscript}
          />
        </div>
      </div>

      {/* Step 2: Self Assessment */}
      {selectedTranscript && (
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              currentStep === 'complete' ? 'bg-primary text-primary-foreground' :
              currentStep === 'assessment' ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
            }`}>
              {assessment ? '✓' : '2'}
            </div>
            <h2 className="text-lg font-semibold">Self-Reflection</h2>
          </div>

          <div className={currentStep === 'transcript' ? 'opacity-60' : ''}>
            <SelfAssessmentTable
              onAssessmentComplete={handleAssessmentComplete}
              selectedTranscript={selectedTranscript.title}
            />
          </div>
        </div>
      )}

      {/* Step 3: Ready to Start */}
      {currentStep === 'complete' && selectedTranscript && assessment && (
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-medium">
              ✓
            </div>
            <h2 className="text-lg font-semibold">Ready for Coaching</h2>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="text-green-800 mb-3">
              <h3 className="font-medium mb-2">Setup Complete!</h3>
              <p className="text-sm">
                You've selected "<span className="font-medium">{selectedTranscript.title}</span>"
                and completed your self-reflection. You can now connect to begin your coaching session.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleStartOver}
                className="px-4 py-2 text-sm border border-green-300 text-green-700 rounded-md hover:bg-green-100 transition-colors"
              >
                Start Over
              </button>
              <div className="text-sm text-green-600 flex items-center">
                → Ready to connect to your AI coach
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CoachingSetup;