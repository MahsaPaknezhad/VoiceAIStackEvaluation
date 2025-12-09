import React, { useState } from 'react';
import CoachingSetup from './CoachingSetup';
import { type Transcript } from '../data/transcripts';

interface SelfAssessmentData {
  wentWell: string;
  improvements: string;
}

const TestPage: React.FC = () => {
  const [setupData, setSetupData] = useState<{
    transcript: Transcript;
    assessment: SelfAssessmentData;
  } | null>(null);

  const handleSetupComplete = (data: { transcript: Transcript; assessment: SelfAssessmentData }) => {
    setSetupData(data);
    console.log('Setup completed with data:', data);
  };

  const handleReset = () => {
    setSetupData(null);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-card border-b px-4 py-3">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src="/nab_logo.png"
              alt="NAB"
              className="w-8 h-8"
            />
            <h1 className="text-lg font-semibold">NAB AI Coach - Component Test</h1>
          </div>
          {setupData && (
            <button
              onClick={handleReset}
              className="px-3 py-1 text-sm bg-muted text-muted-foreground rounded hover:bg-muted/80"
            >
              Reset Test
            </button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {!setupData ? (
          <CoachingSetup onSetupComplete={handleSetupComplete} />
        ) : (
          <div className="max-w-4xl mx-auto">
            <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
              <h2 className="text-lg font-semibold text-green-800 mb-4">
                ✅ Setup Flow Complete!
              </h2>

              <div className="space-y-4 text-sm">
                <div>
                  <span className="font-medium text-green-700">Selected Transcript:</span>
                  <div className="mt-1 p-3 bg-white rounded border">
                    <div className="font-medium">{setupData.transcript.title}</div>
                    <div className="text-muted-foreground text-xs mt-1">
                      {setupData.transcript.description}
                    </div>
                    <div className="text-xs text-muted-foreground mt-2">
                      Participants: {setupData.transcript.participants.join(', ')}
                    </div>
                  </div>
                </div>

                <div>
                  <span className="font-medium text-green-700">Self-Reflection:</span>
                  <div className="mt-1 space-y-2">
                    <div className="p-3 bg-white rounded border">
                      <div className="font-medium text-xs text-muted-foreground mb-1">
                        What went well:
                      </div>
                      <div>{setupData.assessment.wentWell}</div>
                    </div>
                    <div className="p-3 bg-white rounded border">
                      <div className="font-medium text-xs text-muted-foreground mb-1">
                        What could be improved:
                      </div>
                      <div>{setupData.assessment.improvements}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-green-200">
                <div className="text-green-700 text-sm">
                  💡 <strong>Next step:</strong> This data would now be passed to the AI coaching bot
                  to start the guided conversation.
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TestPage;