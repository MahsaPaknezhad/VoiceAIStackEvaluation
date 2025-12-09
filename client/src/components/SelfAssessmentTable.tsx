import React, { useState } from 'react';

interface SelfAssessmentData {
  doRightThing: Record<string, 'great' | 'good' | 'improvement' | null>;
  customerEngagement: Record<string, 'great' | 'good' | 'improvement' | null>;
}

interface SelfAssessmentTableProps {
  onAssessmentComplete: (assessment: SelfAssessmentData) => void;
  selectedTranscript: string | null;
}

const SelfAssessmentTable: React.FC<SelfAssessmentTableProps> = ({
  onAssessmentComplete,
  selectedTranscript
}) => {
  const [assessment, setAssessment] = useState<SelfAssessmentData>({
    doRightThing: {
      'customer-experience-centre': null,
      'understand-deeper-level': null,
      'respected-customer-time': null,
    },
    customerEngagement: {
      'explained-purpose': null,
      'listened-without-limiting': null,
      'showed-interest': null,
      'demonstrated-appreciation': null,
      'ended-with-summary': null,
    }
  });

  const updateRating = (section: keyof SelfAssessmentData, criterion: string, rating: 'great' | 'good' | 'improvement') => {
    setAssessment(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [criterion]: rating
      }
    }));
  };

  const isComplete = () => {
    const allRightThing = Object.values(assessment.doRightThing).every(val => val !== null);
    const allEngagement = Object.values(assessment.customerEngagement).every(val => val !== null);
    return allRightThing && allEngagement;
  };

  const handleSubmit = () => {
    if (isComplete()) {
      onAssessmentComplete(assessment);
    }
  };

  const RatingButtons = ({ section, criterion, value }: {
    section: keyof SelfAssessmentData;
    criterion: string;
    value: 'great' | 'good' | 'improvement' | null;
  }) => (
    <div className="flex gap-2">
      {(['great', 'good', 'improvement'] as const).map((rating) => (
        <button
          key={rating}
          onClick={() => updateRating(section, criterion, rating)}
          className={`px-3 py-1 text-xs rounded-md border transition-colors ${
            value === rating
              ? rating === 'great'
                ? 'bg-green-100 border-green-500 text-green-800'
                : rating === 'good'
                ? 'bg-blue-100 border-blue-500 text-blue-800'
                : 'bg-orange-100 border-orange-500 text-orange-800'
              : 'bg-background border-border text-muted-foreground hover:border-muted-foreground'
          }`}
        >
          {rating === 'great' && 'Great'}
          {rating === 'good' && 'Good'}
          {rating === 'improvement' && 'Needs Improvement'}
        </button>
      ))}
    </div>
  );

  return (
    <div className="w-full max-w-4xl">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-foreground mb-2">
          📝 Self-Reflection Rubric
        </h3>
        <p className="text-sm text-muted-foreground">
          Rate your performance on the selected transcript: <span className="font-medium">{selectedTranscript}</span>
        </p>
      </div>

      <div className="bg-card border border-border rounded-lg overflow-hidden">

        {/* Do the right thing section */}
        <div className="p-4 border-b border-border">
          <h4 className="font-semibold text-red-600 mb-3">Do the right thing (the "what")</h4>
          <div className="space-y-3">
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Keep customers experience and expectations at centre of the conversation</div>
              <div className="col-span-3">
                <RatingButtons section="doRightThing" criterion="customer-experience-centre" value={assessment.doRightThing['customer-experience-centre']} />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Sought to understand and learn more at a deeper level the experience and impact(s)</div>
              <div className="col-span-3">
                <RatingButtons section="doRightThing" criterion="understand-deeper-level" value={assessment.doRightThing['understand-deeper-level']} />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Respected customer time by not over-explaining/over-asking</div>
              <div className="col-span-3">
                <RatingButtons section="doRightThing" criterion="respected-customer-time" value={assessment.doRightThing['respected-customer-time']} />
              </div>
            </div>
          </div>
        </div>

        {/* Customer engagement section */}
        <div className="p-4">
          <h4 className="font-semibold text-red-600 mb-3">Customer engagement. Empathy. (the "how")</h4>
          <div className="space-y-3">
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Explained purpose of call as learning from their feedback</div>
              <div className="col-span-3">
                <RatingButtons section="customerEngagement" criterion="explained-purpose" value={assessment.customerEngagement['explained-purpose']} />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Listened without limiting answers. Asked questions that encouraged customer to reflect on experience and respond candidly</div>
              <div className="col-span-3">
                <RatingButtons section="customerEngagement" criterion="listened-without-limiting" value={assessment.customerEngagement['listened-without-limiting']} />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Showed interest in responses. Asked clarifying and follow-up questions to understand experience, expectations and any improvement suggestions</div>
              <div className="col-span-3">
                <RatingButtons section="customerEngagement" criterion="showed-interest" value={assessment.customerEngagement['showed-interest']} />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Demonstrated appreciation of customer perspective and commitment to improve their experience. Apologised and solved discrete issues as appropriate</div>
              <div className="col-span-3">
                <RatingButtons section="customerEngagement" criterion="demonstrated-appreciation" value={assessment.customerEngagement['demonstrated-appreciation']} />
              </div>
            </div>
            <div className="grid grid-cols-4 gap-4 items-center py-2">
              <div className="text-sm">Ended call with a summary of main points to confirm understanding</div>
              <div className="col-span-3">
                <RatingButtons section="customerEngagement" criterion="ended-with-summary" value={assessment.customerEngagement['ended-with-summary']} />
              </div>
            </div>
          </div>
        </div>

        {/* Submit button */}
        <div className="p-4 border-t border-border bg-muted/25">
          <div className="flex justify-end">
            <button
              onClick={handleSubmit}
              disabled={!isComplete()}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                isComplete()
                  ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                  : 'bg-muted text-muted-foreground cursor-not-allowed'
              }`}
            >
              Continue to Coaching Session
            </button>
          </div>
        </div>
      </div>

      {/* Progress indicator */}
      <div className="mt-4 text-center">
        <div className="text-xs text-muted-foreground">
          Step 2 of 3: Complete your self-reflection
        </div>
        <div className="w-full bg-muted rounded-full h-1.5 mt-2">
          <div
            className="bg-primary h-1.5 rounded-full transition-all duration-300"
            style={{ width: isComplete() ? '66%' : '33%' }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default SelfAssessmentTable;