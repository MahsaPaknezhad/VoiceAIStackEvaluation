import React, { useState } from 'react';
import CoachingSetup from './CoachingSetup';
import { type Transcript } from '../data/transcripts';

interface SelfAssessmentData {
  doRightThing: Record<string, 'great' | 'good' | 'improvement' | null>;
  customerEngagement: Record<string, 'great' | 'good' | 'improvement' | null>;
}

interface SetupData {
  transcript: Transcript;
  assessment: SelfAssessmentData;
}

// Mock Chat Component (replicates the real chat interface)
const MockChatInterface: React.FC<{ setupData: SetupData }> = ({ setupData }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant' as const,
      content: `Hi! I can see you've selected the transcript "${setupData.transcript.title}" and completed your self-reflection. Let's work together to improve your customer interaction skills. What would you like to focus on first?`
    }
  ]);
  const [inputValue, setInputValue] = useState('');

  const addMessage = () => {
    if (inputValue.trim()) {
      setMessages(prev => [...prev,
        { role: 'user' as const, content: inputValue },
        { role: 'assistant' as const, content: 'Thanks for sharing that. Based on your self-reflection, I notice you rated some areas for improvement. Let\'s explore those together...' }
      ]);
      setInputValue('');
    }
  };

  const getAssessmentSummary = () => {
    const allRatings = [...Object.values(setupData.assessment.doRightThing), ...Object.values(setupData.assessment.customerEngagement)];
    const great = allRatings.filter(r => r === 'great').length;
    const good = allRatings.filter(r => r === 'good').length;
    const improvement = allRatings.filter(r => r === 'improvement').length;
    return { great, good, improvement };
  };

  const summary = getAssessmentSummary();

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <div className="flex-shrink-0 border-b bg-card/50">
        <div className="px-3 py-2 sm:px-4 sm:py-3">
          <div className="max-w-6xl mx-auto flex items-center justify-between gap-2 sm:gap-4">
            {/* Left: Logo, Title */}
            <div className="flex items-center gap-3 sm:gap-4">
              <img
                src="/nab_logo.png"
                alt="NAB"
                className="w-8 h-8 sm:w-9 sm:h-9"
              />
              <div className="flex flex-col items-start">
                <h1 className="text-base sm:text-lg font-semibold leading-tight">NAB AI Coach</h1>
                <div className="text-xs text-muted-foreground">
                  Full Flow Test - With Setup Integration
                </div>
              </div>
            </div>

            {/* Right: Mock Controls */}
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                Connected
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Session Info Header */}
      <div className="flex-shrink-0 bg-blue-50 border-b border-blue-200 px-4 py-2">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              <div>
                <span className="font-medium text-blue-800">📋 Transcript:</span>
                <span className="ml-1 text-blue-700">{setupData.transcript.title}</span>
              </div>
              <div>
                <span className="font-medium text-blue-800">📊 Reflection:</span>
                <span className="ml-1 text-blue-700">
                  {summary.great} Great, {summary.good} Good, {summary.improvement} Needs Improvement
                </span>
              </div>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="text-xs text-blue-600 hover:text-blue-800"
            >
              Start Over
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0 p-2 sm:p-3">
        <div className="max-w-7xl mx-auto h-full flex gap-4">

          {/* Left Sidebar - Reference Panels */}
          <div className="w-80 flex flex-col gap-4">

            {/* Transcript Reference */}
            <div className="bg-card rounded-lg border">
              <div className="px-3 py-2 border-b bg-muted/50">
                <h3 className="text-sm font-semibold">📋 Selected Transcript</h3>
              </div>
              <div className="p-3 text-xs">
                <div className="font-medium mb-1">{setupData.transcript.title}</div>
                <div className="text-muted-foreground mb-2">{setupData.transcript.description}</div>
                <div className="text-muted-foreground">
                  <span className="font-medium">Participants:</span> {setupData.transcript.participants.join(', ')}
                </div>
                <div className="text-muted-foreground mt-1">
                  <span className="font-medium">Duration:</span> {setupData.transcript.duration}
                </div>
              </div>
            </div>

            {/* Assessment Reference */}
            <div className="bg-card rounded-lg border flex-1">
              <div className="px-3 py-2 border-b bg-muted/50">
                <h3 className="text-sm font-semibold">📊 Your Self-Reflection</h3>
              </div>
              <div className="p-3 text-xs space-y-3 max-h-96 overflow-y-auto">

                {/* Do the right thing section */}
                <div>
                  <h4 className="font-semibold text-red-600 mb-2 text-xs">Do the right thing</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Customer experience centre</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.doRightThing['customer-experience-centre'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.doRightThing['customer-experience-centre'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.doRightThing['customer-experience-centre'] === 'great' ? 'Great' :
                         setupData.assessment.doRightThing['customer-experience-centre'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Understand deeper level</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.doRightThing['understand-deeper-level'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.doRightThing['understand-deeper-level'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.doRightThing['understand-deeper-level'] === 'great' ? 'Great' :
                         setupData.assessment.doRightThing['understand-deeper-level'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Respected customer time</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.doRightThing['respected-customer-time'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.doRightThing['respected-customer-time'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.doRightThing['respected-customer-time'] === 'great' ? 'Great' :
                         setupData.assessment.doRightThing['respected-customer-time'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Customer engagement section */}
                <div>
                  <h4 className="font-semibold text-red-600 mb-2 text-xs">Customer engagement</h4>
                  <div className="space-y-1">
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Explained purpose</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.customerEngagement['explained-purpose'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.customerEngagement['explained-purpose'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.customerEngagement['explained-purpose'] === 'great' ? 'Great' :
                         setupData.assessment.customerEngagement['explained-purpose'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Listened without limiting</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.customerEngagement['listened-without-limiting'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.customerEngagement['listened-without-limiting'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.customerEngagement['listened-without-limiting'] === 'great' ? 'Great' :
                         setupData.assessment.customerEngagement['listened-without-limiting'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Showed interest</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.customerEngagement['showed-interest'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.customerEngagement['showed-interest'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.customerEngagement['showed-interest'] === 'great' ? 'Great' :
                         setupData.assessment.customerEngagement['showed-interest'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Demonstrated appreciation</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.customerEngagement['demonstrated-appreciation'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.customerEngagement['demonstrated-appreciation'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.customerEngagement['demonstrated-appreciation'] === 'great' ? 'Great' :
                         setupData.assessment.customerEngagement['demonstrated-appreciation'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                    <div className="flex justify-between items-start gap-2">
                      <span className="text-muted-foreground">Ended with summary</span>
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                        setupData.assessment.customerEngagement['ended-with-summary'] === 'great' ? 'bg-green-100 text-green-800' :
                        setupData.assessment.customerEngagement['ended-with-summary'] === 'good' ? 'bg-blue-100 text-blue-800' :
                        'bg-orange-100 text-orange-800'
                      }`}>
                        {setupData.assessment.customerEngagement['ended-with-summary'] === 'great' ? 'Great' :
                         setupData.assessment.customerEngagement['ended-with-summary'] === 'good' ? 'Good' : 'Needs Improvement'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Side - Chat Interface */}
          <div className="flex-1 bg-card rounded-lg border flex flex-col">
            {/* Conversation Header */}
            <div className="flex-shrink-0 px-3 py-2 sm:px-4 sm:py-3 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-sm sm:text-base font-bold">Coaching Conversation</h2>
                <div className="flex items-center gap-2 text-xs">
                  <div className="h-3 w-3 rounded-full bg-green-500"></div>
                  <span className="text-muted-foreground">Ready</span>
                </div>
              </div>
            </div>

            {/* Chat Area */}
            <div className="flex-1 min-h-0 p-2 sm:p-3">
              <div className="h-full overflow-y-auto bg-muted/10 rounded-md p-2 sm:p-3">
                <div className="space-y-6 sm:space-y-8 max-w-none">
                  {messages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${
                        message.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-[85%] px-2.5 py-1.5 sm:px-3 sm:py-2 rounded-md text-sm ${
                          message.role === 'user'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-background text-foreground border border-border'
                        }`}
                      >
                        <p className="leading-relaxed">{message.content}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Input Area */}
            <div className="flex-shrink-0 p-3 border-t bg-muted/25">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && addMessage()}
                  placeholder="Type your response or speak..."
                  className="flex-1 px-3 py-2 bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary text-sm"
                />
                <button
                  onClick={addMessage}
                  disabled={!inputValue.trim()}
                  className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const FullFlowTestPage: React.FC = () => {
  const [setupData, setSetupData] = useState<SetupData | null>(null);

  const handleSetupComplete = (data: SetupData) => {
    setSetupData(data);
    console.log('Setup completed, transitioning to chat with data:', data);
  };

  // Show setup flow first
  if (!setupData) {
    return (
      <div className="min-h-screen bg-background">
        <CoachingSetup onSetupComplete={handleSetupComplete} />
      </div>
    );
  }

  // Show chat interface with setup data
  return <MockChatInterface setupData={setupData} />;
};

export default FullFlowTestPage;