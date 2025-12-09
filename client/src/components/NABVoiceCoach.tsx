import React, { useState, useEffect, useRef } from 'react';
import {
  PipecatAppBase,
  FullScreenContainer,
  usePipecatConversation,
  usePipecatEventStream,
  ConnectButton,
  UserAudioControl,
  ErrorCard,
  SpinLoader,
  type PipecatBaseChildProps,
} from "@pipecat-ai/voice-ui-kit";

interface NABVoiceCoachProps {
  connectParams: {
    webrtcRequestParams?: {
      endpoint: string;
    };
  };
}


interface SelfReflectionData {
  rubric: {
    stars: number;
    expectation?: string;
    observations?: string;
  }[];
}

// Removed FlowStage as we're using a side-by-side layout now

// AI Status Indicator Component
const AIStatusIndicator: React.FC<{ client?: any }> = ({ client }) => {
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [isAISpeaking, setIsAISpeaking] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  
  // Try to listen to RTVI events, with fallback if not available
  let events: readonly any[] = [];
  try {
    const eventStream = usePipecatEventStream();
    events = eventStream?.events || [];
  } catch (error) {
    console.log('RTVI events not available, using fallback');
  }
  
  // Monitor client connection state
  useEffect(() => {
    if (!client) {
      console.log('🔌 Client disconnected');
      setConnectionState('disconnected');
      return;
    }
    
    console.log('🔌 Client connected:', client);
    setConnectionState('connected');
    
    // Listen for connection events if available
    const handleConnect = () => {
      console.log('🔌 Client connect event');
      setConnectionState('connected');
    };
    const handleDisconnect = () => {
      console.log('🔌 Client disconnect event');
      setConnectionState('disconnected');
    };
    const handleConnecting = () => {
      console.log('🔌 Client connecting event');
      setConnectionState('connecting');
    };
    
    if (client.on) {
      client.on('connect', handleConnect);
      client.on('disconnect', handleDisconnect);
      client.on('connecting', handleConnecting);
      
      return () => {
        client.off('connect', handleConnect);
        client.off('disconnect', handleDisconnect);
        client.off('connecting', handleConnecting);
      };
    }
  }, [client]);
  
  // Monitor RTVI events for speaking/listening states
  useEffect(() => {
    if (!events || events.length === 0) return;
    
    const latestEvent = events[events.length - 1];
    
    // Log all RTVI events to console for debugging
    console.log(latestEvent.type, {
      type: latestEvent.type,
      timestamp: new Date().toISOString(),
      event: latestEvent,
      totalEvents: events.length
    });
    
    switch (latestEvent.type) {
      case 'botTtsText':
        console.log('🤖 AI Started Speaking');
        setIsAISpeaking(true);
        setIsUserSpeaking(false);
        break;
      case 'botStoppedSpeaking':
        console.log('🤖 AI Stopped Speaking');
        setIsAISpeaking(false);
        break;
      case 'userStartedSpeaking':
        console.log('👤 User Started Speaking (AI Listening)');
        setIsUserSpeaking(true);
        setIsAISpeaking(false);
        break;
      case 'userStoppedSpeaking':
        console.log('👤 User Stopped Speaking');
        setIsUserSpeaking(false);
        break;
      case 'serverMessage':
        if (latestEvent.data.type === 'eval-ready') {
          console.log(latestEvent.data.payload);
          // TODO: Display eval rubric
        }
        else {
          console.log('🤖 Unknown server message');
        }
        break;
    }
  }, [events]);
  
  const getStatusInfo = () => {
    if (connectionState === 'disconnected') {
      return {
        text: 'Offline',
        color: 'bg-gray-400',
        icon: '💤',
        animation: 'animate-pulse'
      };
    }
    
    if (connectionState === 'connecting') {
      return {
        text: 'Connecting',
        color: 'bg-yellow-500',
        icon: '⏳',
        animation: 'animate-pulse'
      };
    }
    
    if (connectionState === 'connected') {
      if (isAISpeaking) {
        return {
          text: 'Speaking',
          color: 'bg-blue-500',
          icon: '🤖',
          animation: 'ai-status-speaking'
        };
      }
      
      if (isUserSpeaking) {
        return {
          text: 'Listening',
          color: 'bg-orange-500',
          icon: '👤',
          animation: 'ai-status-listening'
        };
      }
      
      return {
        text: 'Ready',
        color: 'bg-green-500',
        icon: '👤',
        animation: 'ai-status-listening'
      };
    }
    
    return {
      text: 'Ready',
      color: 'bg-blue-500',
      icon: '👤',
      animation: 'animate-pulse'
    };
  };

  const status = getStatusInfo();

  // Log current status for debugging
  console.log('📊 AI Status Indicator State:', {
    connectionState,
    isAISpeaking,
    isUserSpeaking,
    status: status.text,
    color: status.color,
    eventsLength: events.length
  });

  return (
    <div className="flex items-center gap-2 text-xs" style={{ minWidth: '80px' }}>
      <div 
        className={`h-3 w-3 rounded-full ${status.color} ${status.animation}`}
        style={{ display: 'block' }}
      ></div>
      <span className="text-muted-foreground text-xs font-medium">{status.text}</span>
      <span className="text-base" style={{ display: 'inline-block' }}>{status.icon}</span>
    </div>
  );
};


// Left Control Panel Component (combines transcript selection and self-review)
const LeftControlPanel: React.FC<{
  selectedTranscript: string;
  onTranscriptChange: (transcript: string) => void;
  availableTranscripts: string[];
  loadingTranscripts: boolean;
  selfReviewData: SelfReflectionData;
  onSelfReviewChange: (reviewData: SelfReflectionData) => void;
  evaluationResults: SelfReflectionData | null;
  onConnect: () => void;
  isConnected: boolean;
}> = ({ 
  selectedTranscript, 
  onTranscriptChange, 
  availableTranscripts, 
  loadingTranscripts,
  selfReviewData, 
  onSelfReviewChange,
  evaluationResults,
  onConnect,
  isConnected
}) => {
  const updateStarRating = (index: number, stars: number) => {
    const updatedRubric = [...selfReviewData.rubric];
    updatedRubric[index] = { stars };
    onSelfReviewChange({ rubric: updatedRubric });
  };

  const criteriaLabels = [
    "Customer Experience Focus",
    "Deep Understanding",
    "Respected Customer Time", 
    "Clear Purpose Communication",
    "Active Listening",
    "Interest & Follow-up",
    "Appreciation & Commitment",
    "Call Summary"
  ];

  return (
    <div className="h-full flex flex-col p-4 overflow-hidden">
      {/* Transcript Selection */}
      <div className="flex-shrink-0 mb-16" style={{ position: 'relative', zIndex: 1001 }}>
        <h3 className="text-base font-semibold mb-3">Select Transcript</h3>
        
        <select
          value={selectedTranscript}
          onChange={(e) => {
            console.log('🎯 Dropdown changed to:', e.target.value);
            onTranscriptChange(e.target.value);
          }}
          className="w-full px-3 py-2 border border-border rounded-md text-sm bg-muted text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          style={{ zIndex: 1002 }}
        >
          <option value="">Choose a transcript...</option>
          {availableTranscripts.map((transcript, index) => {
            console.log(`🔍 Rendering option ${index}:`, transcript);
            return (
              <option key={`transcript-${index}`} value={transcript}>
                {transcript.replace('.txt', '')}
              </option>
            );
          })}
        </select>
      </div>

      {/* Self Review */}
      <div style={{ height: '20px' }} />
      <div className="flex-1 flex flex-col min-h-0 relative z-20 pt-8">
        <h3 className="text-base font-semibold mb-3">Self-Review & Coach Feedback</h3>
        <div className="flex-1 overflow-y-auto space-y-3 mb-4 relative z-30">
          {selfReviewData.rubric.map((item, index) => (
            <div key={index} className="p-3 border border-border rounded-md bg-background shadow-sm hover:shadow-md transition-shadow relative z-40 break-words overflow-hidden">
              {/* Header with criteria name */}
              <div className="mb-2">
                <h4 className="font-semibold text-sm text-foreground leading-tight">
                  {criteriaLabels[index] || `Criteria ${index + 1}`}
                </h4>
              </div>
              
              {/* Compact Ratings Row */}
              <div className="grid grid-cols-2 gap-3 mb-2">
                {/* Your Rating - Compact */}
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                    Your Rating
                  </label>
                  <div className="space-y-2">
                    <div className="w-1/2">
                      <input
                        type="range"
                        min="1"
                        max="3"
                        value={item.stars}
                        onChange={(e) => {
                          updateStarRating(index, parseInt(e.target.value));
                        }}
                        className="w-full h-1 bg-gray-100 rounded-lg appearance-none cursor-pointer"
                        style={{
                          background: `linear-gradient(to right, #9ca3af 0%, #9ca3af ${((item.stars - 1) / 2) * 100}%, #f3f4f6 ${((item.stars - 1) / 2) * 100}%, #f3f4f6 100%)`,
                          accentColor: '#9ca3af'
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-muted-foreground w-1/2">
                      <span className={item.stars === 1 ? 'text-gray-600 font-medium' : ''}>1★</span>
                      <span className={item.stars === 2 ? 'text-gray-600 font-medium' : ''}>2★</span>
                      <span className={item.stars === 3 ? 'text-gray-600 font-medium' : ''}>3★</span>
                    </div>
                  </div>
                </div>

                {/* Coach Rating - Compact */}
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                    Coach Rating
                  </label>
                  <div className="flex items-center gap-0.5 pt-1">
                    {evaluationResults && evaluationResults.rubric[index] ? (
                      (() => {
                        const stars = evaluationResults.rubric[index].stars || 0;
                        if (stars <= 0) {
                          return (
                            <span className="text-xs text-muted-foreground italic">
                              Pending
                            </span>
                          );
                        }
                        return (
                          <>
                            {Array.from({ length: stars }).map((_, i) => (
                              <span key={i} className="text-sm leading-none text-blue-500">★</span>
                            ))}
                          </>
                        );
                      })()
                    ) : (
                      <span className="text-xs text-muted-foreground italic">
                        Pending
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Coach Observations - Compact */}
              {(evaluationResults && evaluationResults.rubric[index]?.observations) ? (
                <div className="mt-2 pt-2 border-t border-border/50">
                  <label className="text-xs font-medium text-muted-foreground block mb-1">
                    Coach Notes
                  </label>
                  <p className="text-xs text-foreground bg-blue-50 dark:bg-blue-950/20 p-2 rounded text-left leading-snug break-words overflow-wrap-anywhere">
                    {evaluationResults.rubric[index].observations}
                  </p>
                </div>
              ) : (
                <div className="mt-2 pt-2 border-t border-border/20">
                  <label className="text-xs font-medium text-muted-foreground block mb-1">
                    Coach Notes
                  </label>
                  <p className="text-xs text-muted-foreground/70 italic">
                    Pending
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};


// Voice Chat Area Component (right side)
const VoiceChatArea: React.FC<{
  client?: any;
}> = ({ client }) => {
  const { messages } = usePipecatConversation();
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    // Prefer scrolling a sentinel into view for reliable behavior
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      return;
    }
    // Fallback: direct scrollTop manipulation
    if (scrollContainerRef.current) {
      const el = scrollContainerRef.current;
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="h-full flex flex-col bg-card">
      {/* Chat Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b bg-muted/5">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">AI Coach Session</h3>
          <AIStatusIndicator client={client} />
        </div>
      </div>

      {/* Chat Messages */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto p-4 conversation-scroll">
          <div className="max-w-none">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex bubble-row ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
              <div
                className={`max-w-[80%] rounded-lg text-sm ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-foreground'
                } shadow-sm`}
              >
                <div className="bubble-inner">
                  <p className="leading-relaxed whitespace-pre-wrap m-0">{message.content}</p>
                </div>
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
        
        {messages.length === 0 && (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-muted-foreground max-w-md">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted/30 flex items-center justify-center">
                <div className="text-2xl">🎙️</div>
              </div>
              <p className="text-sm text-foreground">Connect to begin session</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};



// Custom NAB UI Component
const NABCustomUI: React.FC<PipecatBaseChildProps> = ({
  client,
  handleConnect,
  handleDisconnect,
  error
}) => {
  const [selectedTranscript, setSelectedTranscript] = useState<string>('');
  const [availableTranscripts, setAvailableTranscripts] = useState<string[]>([]);
  const [loadingTranscripts, setLoadingTranscripts] = useState<boolean>(true);
  
  // Debug log for availableTranscripts changes
  useEffect(() => {
    console.log('📋 availableTranscripts state updated:', availableTranscripts);
    console.log('📋 loadingTranscripts:', loadingTranscripts);
  }, [availableTranscripts, loadingTranscripts]);
  const [selfReviewData, setSelfReviewData] = useState<SelfReflectionData>({
    rubric: Array.from({ length: 8 }, () => ({ stars: 2 })) // Default 2 stars for each criteria (1-3 scale)
  });
  const [evaluationResults, setEvaluationResults] = useState<SelfReflectionData | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState<number>(33); // Percentage
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Ensure mic devices are initialized so mute/unmute works regardless of connection
  useEffect(() => {
    if (!client) return;
    try {
      if (typeof client.initDevices === 'function') {
        client.initDevices();
      }
    } catch (err) {
      console.error('Failed to init devices:', err);
    }
  }, [client]);

  // Handle resize drag
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;
    
    const containerWidth = window.innerWidth;
    const newWidth = (e.clientX / containerWidth) * 100;
    
    // Constrain between 20% and 60%
    const constrainedWidth = Math.min(Math.max(newWidth, 20), 60);
    setLeftPanelWidth(constrainedWidth);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Add event listeners for mouse move and up
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging]);

  // Load available transcripts on component mount
  useEffect(() => {
    console.log('🔍 NABCustomUI component mounted, loading transcripts...');
    console.log('🔍 Current availableTranscripts state:', availableTranscripts);
    
    const loadTranscripts = async () => {
      try {
        setLoadingTranscripts(true);
        console.log('📡 Starting fetch to /api/transcripts');
        
        const response = await fetch('/api/transcripts');
        console.log('📥 Transcript response status:', response.status);
        console.log('📥 Response ok:', response.ok);
        console.log('📥 Response url:', response.url);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📄 Transcript data received:', data);
        console.log('📄 Available transcripts array:', data.transcripts);
        console.log('📄 Setting availableTranscripts to:', data.transcripts || []);
        
        setAvailableTranscripts(data.transcripts || []);
        console.log('✅ Successfully set availableTranscripts');
        
      } catch (error) {
        console.error('❌ Error loading transcript list:', error);
        console.error('❌ Error details:', error instanceof Error ? error.message : 'Unknown error');
        setAvailableTranscripts([]);
      } finally {
        setLoadingTranscripts(false);
        console.log('🏁 Finished loading transcripts');
      }
    };

    loadTranscripts();
  }, []);


  // Monitor connection state
  useEffect(() => {
    console.log('📊 Client state changed:', !!client, 'client:', client);
    setIsConnected(!!client);
  }, [client]);

  // Listen for evaluation results from the bot
  const { events } = usePipecatEventStream();
  useEffect(() => {
    if (!events || events.length === 0) return;
    
    const latestEvent = events[events.length - 1];
    
    if (latestEvent.type === 'serverMessage' && latestEvent.data && 
        typeof latestEvent.data === 'object' && 
        'type' in latestEvent.data && 
        latestEvent.data.type === 'eval-ready') {
      console.log('Received evaluation results:', latestEvent.data);
      const data = latestEvent.data as { type: string; payload: SelfReflectionData };
      setEvaluationResults(data.payload);
    }
  }, [events]);

  // Enhanced connect handler that sends transcript and self-reflection data
  const handleConnectWithData = async () => {
    console.log('🔥 handleConnectWithData called');
    console.log('🔍 handleConnect function:', handleConnect);
    console.log('🔍 handleConnect type:', typeof handleConnect);
    console.log('🔍 selectedTranscript:', selectedTranscript);
    console.log('🔍 availableTranscripts:', availableTranscripts);
    
    if (!selectedTranscript) {
      console.error('❌ No transcript selected');
      alert('Please select a transcript first');
      return;
    }

    try {
      console.log('📞 Fetching transcript content...');
      // Load selected transcript content
      const transcriptResponse = await fetch(`/api/transcripts/${selectedTranscript}`);
      const transcriptData = await transcriptResponse.json();
      console.log('✅ Transcript loaded:', transcriptData);
      
      // Prepare connection data
      const connectionData = {
        transcript: transcriptData.content,
        transcriptName: selectedTranscript,
        selfReflection: selfReviewData
      };

      console.log('📦 Connection data prepared:');
      console.log('📦 - transcriptName:', connectionData.transcriptName);
      console.log('📦 - transcript length:', connectionData.transcript.length);
      console.log('📦 - selfReflection:', connectionData.selfReflection);
      
      console.log('📡 Sending client data to server...');
      // Send client data to server before WebRTC connection
      const clientDataResponse = await fetch('/api/set-client-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(connectionData),
      });
      
      if (!clientDataResponse.ok) {
        throw new Error(`Failed to set client data: ${clientDataResponse.status}`);
      }
      
      const clientDataResult = await clientDataResponse.json();
      console.log('✅ Client data set on server:', clientDataResult);
      
      console.log('🚀 Calling handleConnect...');
      if (handleConnect) {
        await handleConnect();
        console.log('✅ handleConnect completed');
      } else {
        console.error('❌ handleConnect is undefined!');
        alert('Connection handler not available');
      }
  } catch (error) {
      console.error('💥 Error connecting with data:', error);
      const message = error instanceof Error ? error.message : 'Unknown error';
      alert(`Connection error: ${message}`);
      // Fallback to original connect handler
      if (handleConnect) {
        console.log('🔄 Attempting fallback connection...');
        await handleConnect();
      }
    }
  };

  if (!client) {
    return (
      <FullScreenContainer>
        <div className="flex items-center justify-center h-full">
          <SpinLoader />
        </div>
      </FullScreenContainer>
    );
  }

  if (error) {
    return (
      <FullScreenContainer>
        <div className="flex items-center justify-center h-full">
          <ErrorCard title="Connection Error">
            {error}
          </ErrorCard>
        </div>
      </FullScreenContainer>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-background" style={{ position: 'relative', zIndex: 1 }}>
      {/* Compact Header */}
      <div className="flex-shrink-0 border-b bg-card/50">
        <div className="px-3 py-2 sm:px-4 sm:py-3">
          <div className="flex items-center justify-between gap-2 sm:gap-4">
            {/* Left: Logo, Title and Status */}
            <div className="flex items-center gap-3 sm:gap-4">
              <img 
                src="/nab_logo.png" 
                alt="NAB" 
                className="w-8 h-8 sm:w-9 sm:h-9"
              />
              <div className="flex flex-col items-start">
                <h1 className="text-base sm:text-lg font-semibold leading-tight">NAB AI Coach</h1>
                <div className="relative text-xs text-muted-foreground">
                  <span className="truncate">Powered by Pipecat & Strands Agents</span>
                  <div className="absolute -left-2 top-1/2 transform -translate-y-1/2 h-1 w-1 rounded-full bg-green-500 status-indicator"></div>
                </div>
              </div>
            </div>

            {/* Right: Connection Status */}
            <div className="flex items-center gap-2">
              <UserAudioControl />
              <div>
                <ConnectButton
                  size="sm"
                  onConnect={() => {
                    console.log('🎯 CONNECT BUTTON CLICKED!');
                    console.log('🎯 About to call handleConnectWithData');
                    console.log('🎯 selectedTranscript at click time:', selectedTranscript);
                    console.log('🎯 isConnected state:', isConnected);
                    console.log('🎯 client object:', client);
                    handleConnectWithData();
                  }}
                  onDisconnect={() => {
                    console.log('🎯 DISCONNECT BUTTON CLICKED!');
                    console.log('🎯 isConnected state:', isConnected);
                    console.log('🎯 client object:', client);
                    handleDisconnect?.();
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Side by Side Layout */}
      <div className="flex-1 min-h-0 p-2 sm:p-3">
        <div className="h-full flex">
            {/* Left Panel - Control Panel (adjustable width) */}
            <div 
              className="flex-shrink-0 overflow-hidden" 
              style={{ 
                width: `${leftPanelWidth}%`,
                position: 'relative', 
                zIndex: 100 
              }}
            >
              <div className="bg-card rounded-lg border h-full overflow-hidden mr-1" style={{ position: 'relative', zIndex: 101 }}>
                <LeftControlPanel
                  selectedTranscript={selectedTranscript}
                  onTranscriptChange={setSelectedTranscript}
                  availableTranscripts={availableTranscripts}
                  loadingTranscripts={loadingTranscripts}
                  selfReviewData={selfReviewData}
                  onSelfReviewChange={setSelfReviewData}
                  evaluationResults={evaluationResults}
                  onConnect={handleConnectWithData}
                  isConnected={isConnected}
                />
              </div>
            </div>

            {/* Resize Handle */}
            <div 
              className={`w-4 flex-shrink-0 cursor-col-resize bg-transparent hover:bg-muted/20 transition-colors ${isDragging ? 'bg-muted/30' : ''}`}
              onMouseDown={handleMouseDown}
              style={{ zIndex: 1000 }}
            >
              <div className="h-full w-full flex items-center justify-center">
                <svg 
                  width="16" 
                  height="16" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                  className="text-muted-foreground/50 hover:text-muted-foreground transition-colors"
                >
                  <circle cx="9" cy="12" r="1"/>
                  <circle cx="9" cy="5" r="1"/>
                  <circle cx="9" cy="19" r="1"/>
                  <circle cx="15" cy="12" r="1"/>
                  <circle cx="15" cy="5" r="1"/>
                  <circle cx="15" cy="19" r="1"/>
                </svg>
              </div>
            </div>

            {/* Right Panel - Voice Chat Area (remaining space) */}
            <div className="flex-1 overflow-hidden">
              <div className="bg-card rounded-lg border h-full overflow-hidden ml-1">
                <VoiceChatArea client={client} />
              </div>
            </div>
        </div>
      </div>
    </div>
  );
};

export const NABVoiceCoach: React.FC<NABVoiceCoachProps> = ({ connectParams }) => {
  return (
    <PipecatAppBase
      connectParams={{
        webrtcRequestParams: {
          endpoint: connectParams.webrtcRequestParams?.endpoint || "/api/offer"
        }
      }}
      transportType="smallwebrtc"
      noThemeProvider
    >
      {(props: PipecatBaseChildProps) => <NABCustomUI {...props} />}
    </PipecatAppBase>
  );
};