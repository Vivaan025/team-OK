'use client';

import { useEffect, useRef, useState } from "react";
import { 
  Sidebar as SidebarIcon, 
  AlertCircle,
  X,
} from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import ModuleControl from "@/components/modulecontrol";
import { PatientModule, TranscriptAccumulator, TranscriptDelta, ClientEvent } from "@/types/module";
import TranscriptView from "@/components/transcriptview";


export default function App() {
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);
  const [events, setEvents] = useState<ClientEvent[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptAccumulator>({});
  const [dataChannel, setDataChannel] = useState<RTCDataChannel | null>(null);
  const [activeModule, setActiveModule] = useState<PatientModule | null>(null);
  const peerConnection = useRef<RTCPeerConnection | null>(null);
  const audioElement = useRef<HTMLAudioElement | null>(null);

  const patientModule: PatientModule = {
    id: '1',
    name: 'Sara Johnson',
    description: 'HIV Patient - Regular Checkup',
    status: 'inactive',
    lastAccessed: new Date('2024-01-27')
  };

  const handleTranscriptDelta = (delta: TranscriptDelta) => {
    setTranscripts(prev => ({
      ...prev,
      [delta.response_id]: {
        text: (prev[delta.response_id]?.text || '') + delta.delta,
        itemId: delta.item_id,
        timestamp: Date.now()
      }
    }));
  };

  async function startSession(): Promise<void> {
    try {
      const tokenResponse = await fetch("/api/token");
      const data: { client_secret: { value: string } } = await tokenResponse.json();
      const EPHEMERAL_KEY = data.client_secret.value;

      const pc = new RTCPeerConnection();
      
      audioElement.current = document.createElement("audio");
      audioElement.current.autoplay = true;
      pc.ontrack = (e: RTCTrackEvent) => {
        if (audioElement.current) {
          audioElement.current.srcObject = e.streams[0];
        }
      };

      const ms = await navigator.mediaDevices.getUserMedia({ audio: true });
      pc.addTrack(ms.getTracks()[0]);

      const dc = pc.createDataChannel("oai-events");
      setDataChannel(dc);

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const baseUrl = "https://api.openai.com/v1/realtime";
      const model = "gpt-4o-realtime-preview-2024-12-17";
      const sdpResponse = await fetch(`${baseUrl}?model=${model}`, {
        method: "POST",
        body: offer.sdp,
        headers: {
          Authorization: `Bearer ${EPHEMERAL_KEY}`,
          "Content-Type": "application/sdp",
        },
      });

      const answer: RTCSessionDescriptionInit = {
        type: "answer",
        sdp: await sdpResponse.text(),
      };
      await pc.setRemoteDescription(answer);

      peerConnection.current = pc;
      setActiveModule({ ...patientModule, status: 'active' });
    } catch (error) {
      console.error("Failed to start session:", error);
      throw error;
    }
  }

  function stopSession(): void {
    if (dataChannel) {
      dataChannel.close();
    }
    if (peerConnection.current) {
      peerConnection.current.close();
    }

    setIsSessionActive(false);
    setDataChannel(null);
    peerConnection.current = null;
    setTranscripts({});
    setActiveModule(null);
  }

//   function sendClientEvent(message: ClientEvent): void {
//     if (dataChannel) {
//       message.event_id = message.event_id || crypto.randomUUID();
//       dataChannel.send(JSON.stringify(message));
//       setEvents((prev) => [message, ...prev]);
//     }
//   }

//   function sendTextMessage(message: string): void {
//     const event: ClientEvent = {
//       type: "conversation.item.create",
//       item: {
//         type: "message",
//         role: "user",
//         content: [{ type: "input_text", text: message }],
//       },
//     };

//     sendClientEvent(event);
//     sendClientEvent({ type: "response.create" });
//   }

  useEffect(() => {
    if (dataChannel) {
      dataChannel.addEventListener("message", (e: MessageEvent) => {
        const data = JSON.parse(e.data);
        
        if (data.type === "response.audio_transcript.delta") {
          handleTranscriptDelta(data);
        }
        
        setEvents((prev) => [data, ...prev]);
      });

      dataChannel.addEventListener("open", () => {
        setIsSessionActive(true);
        setEvents([]);
        setTranscripts({});
      });
    }
  }, [dataChannel]);

  return (
    <div className="min-h-screen w-full bg-gray-50">
      <nav className="h-16 bg-white border-b border-gray-200 fixed top-0 left-0 right-0 z-10">
        <div className="h-full mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <SidebarIcon className="h-6 w-6 text-gray-600" />
            <h1 className="text-xl font-semibold text-gray-900">Medical Console</h1>
          </div>
          {activeModule && (
            <div className="flex items-center space-x-4">
              <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                Active Patient: {activeModule.name}
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={stopSession}
                className="text-red-600 hover:text-red-700"
              >
                <X className="h-4 w-4 mr-2" />
                End Session
              </Button>
            </div>
          )}
        </div>
      </nav>

      <main className="pt-24 pb-6 px-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {isSessionActive && (
            <Alert className="bg-blue-50 border-blue-100">
              <AlertCircle className="h-4 w-4 text-blue-600" />
              <AlertDescription className="text-blue-700 text-sm">
                Active consultation session with {activeModule?.name}. All communications are HIPAA-compliant and encrypted.
              </AlertDescription>
            </Alert>
          )}
          
          {!isSessionActive ? (
            <ModuleControl
              module={patientModule}
              onStart={startSession}
              isSessionActive={isSessionActive}
            />
          ) : (
            <TranscriptView transcripts={transcripts} />
          )}
        </div>
      </main>
    </div>
  );
}