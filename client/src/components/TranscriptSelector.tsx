import React, { useState } from 'react';
import { mockTranscripts, type Transcript } from '../data/transcripts';

interface TranscriptSelectorProps {
  onTranscriptSelect: (transcript: Transcript | null) => void;
  selectedTranscript: Transcript | null;
}

const TranscriptSelector: React.FC<TranscriptSelectorProps> = ({
  onTranscriptSelect,
  selectedTranscript
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleSelect = (transcript: Transcript) => {
    onTranscriptSelect(transcript);
    setIsOpen(false);
  };

  const getTypeIcon = (type: Transcript['type']) => {
    switch (type) {
      case 'coaching_session': return '🎯';
      case 'feedback_call': return '📞';
      case 'good_example': return '⭐';
      default: return '📄';
    }
  };

  const getTypeBadge = (type: Transcript['type']) => {
    switch (type) {
      case 'coaching_session': return 'Coaching';
      case 'feedback_call': return 'Feedback';
      case 'good_example': return 'Best Practice';
      default: return 'Other';
    }
  };

  return (
    <div className="w-full max-w-md">
      <label className="block text-sm font-medium text-foreground mb-2">
        Select a transcript to analyze:
      </label>

      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-3 py-2 text-left bg-background border border-border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {selectedTranscript ? (
                <>
                  <span>{getTypeIcon(selectedTranscript.type)}</span>
                  <span className="truncate">{selectedTranscript.title}</span>
                </>
              ) : (
                <span className="text-muted-foreground">Choose a transcript...</span>
              )}
            </div>
            <svg
              className={`w-5 h-5 text-muted-foreground transition-transform ${
                isOpen ? 'rotate-180' : ''
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </button>

        {isOpen && (
          <div className="absolute z-10 w-full mt-1 bg-background border border-border rounded-md shadow-lg max-h-60 overflow-auto">
            <div className="py-1">
              {mockTranscripts.map((transcript) => (
                <button
                  key={transcript.id}
                  onClick={() => handleSelect(transcript)}
                  className="w-full px-3 py-2 text-left hover:bg-muted focus:bg-muted focus:outline-none"
                >
                  <div className="flex items-start gap-2">
                    <span className="text-lg mt-0.5">{getTypeIcon(transcript.type)}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-sm truncate">{transcript.title}</span>
                        <span className="text-xs text-muted-foreground">({transcript.duration})</span>
                      </div>
                      <p className="text-xs text-muted-foreground line-clamp-2 mb-1">
                        {transcript.description}
                      </p>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          transcript.type === 'good_example'
                            ? 'bg-green-100 text-green-800'
                            : transcript.type === 'coaching_session'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-orange-100 text-orange-800'
                        }`}>
                          {getTypeBadge(transcript.type)}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {transcript.participants.length} participants
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {selectedTranscript && (
        <div className="mt-3 p-3 bg-muted/50 rounded-md">
          <div className="text-sm">
            <div className="font-medium mb-1">Selected: {selectedTranscript.title}</div>
            <div className="text-muted-foreground text-xs mb-2">{selectedTranscript.description}</div>
            <div className="text-xs">
              <span className="font-medium">Participants:</span> {selectedTranscript.participants.join(', ')}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TranscriptSelector;