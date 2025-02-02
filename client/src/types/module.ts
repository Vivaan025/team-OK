// Types
export interface ClientEvent {
    type: string;
    event_id?: string;
    item?: {
      type: string;
      role: string;
      content: Array<{
        type: string;
        text: string;
      }>;
    };
  }
  
 export interface TranscriptDelta {
    type: string;
    event_id: string;
    response_id: string;
    item_id: string;
    output_index: number;
    content_index: number;
    delta: string;
  }
  
export interface TranscriptAccumulator {
    [responseId: string]: {
      text: string;
      itemId: string;
      timestamp: number;
    };
  }
  
export interface PatientModule {
    id: string;
    name: string;
    description: string;
    status: 'active' | 'inactive' | 'completed';
    lastAccessed?: Date;
  }
  