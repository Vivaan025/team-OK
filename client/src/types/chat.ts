export interface Message {
  type: 'user' | 'chunk' | 'complete';
  content: string;
  timestamp: string;
}

export interface OpenAIResponse {
  type: string;
  response?: {
    text?: string;
    modalities?: string[];
  };
  error?: {
    message: string;
    code?: string;
  };
}

export interface ClientSecret {
  value: string;
  expires_at: number;
}

export interface InputAudioTranscription {
  model: string;
}

export interface SessionResponse {
  id: string;
  object: string;
  model: string;
  modalities: string[];
  instructions: string;
  voice: string;
  input_audio_format: string;
  output_audio_format: string;
  input_audio_transcription: InputAudioTranscription;
  turn_detection: null | any;
  tools: any[];
  tool_choice: string;
  temperature: number;
  max_response_output_tokens: number;
  client_secret: ClientSecret;
}