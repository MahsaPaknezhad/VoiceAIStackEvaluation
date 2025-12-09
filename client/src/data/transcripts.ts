// Customer callback transcripts for coaching analysis
export interface Transcript {
  id: string;
  title: string;
  description: string;
  participants: string[];
  duration: string;
}

export const mockTranscripts: Transcript[] = [
  {
    id: 'andrew-samuel-callback',
    title: 'Andrew → Samuel (Service follow-up)',
    description: 'Callback to customer about recent service interaction to gather feedback and ensure satisfaction',
    participants: ['Andrew (NAB Employee)', 'Samuel (Customer)'],
    duration: '3:00'
  },
  {
    id: 'zach-nicole-callback',
    title: 'Zach → Nicole (Wait time feedback)',
    description: 'Callback following customer survey response about long branch wait times',
    participants: ['Zach (Branch Manager)', 'Nicole (Customer)'],
    duration: '1:45'
  },
  {
    id: 'ryan-madison-callback',
    title: 'Ryan → Madison (Dispute resolution follow-up)',
    description: 'Callback to check on customer satisfaction after dispute process was completed',
    participants: ['Ryan (NAB Employee)', 'Madison (Customer)'],
    duration: '3:51'
  },
  {
    id: 'sarah-john-callback',
    title: 'Sarah → John (Home loan inquiry follow-up)',
    description: 'Callback to customer who made home loan inquiry to address remaining questions',
    participants: ['Sarah (Home Loan Specialist)', 'John (Customer)'],
    duration: '2:30'
  },
  {
    id: 'david-emma-callback',
    title: 'David → Emma (Service complaint follow-up)',
    description: 'Callback to customer after online banking complaint to ensure issue resolution',
    participants: ['David (Service Manager)', 'Emma (Customer)'],
    duration: '4:15'
  },
  {
    id: 'lisa-mark-callback',
    title: 'Lisa → Mark (Business account setup follow-up)',
    description: 'Callback to business customer after account setup to ensure satisfaction and address questions',
    participants: ['Lisa (Business Specialist)', 'Mark (Business Owner)'],
    duration: '5:20'
  }
];

// Helper function to get transcript by ID
export const getTranscriptById = (id: string): Transcript | undefined => {
  return mockTranscripts.find(transcript => transcript.id === id);
};