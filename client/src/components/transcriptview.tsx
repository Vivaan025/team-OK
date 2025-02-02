import { TranscriptAccumulator } from "@/types/module";
import { Card } from "./ui/card";
import { MessageSquare } from "lucide-react";
import { Badge } from "./ui/badge";

  function TranscriptView({ transcripts }: { transcripts: TranscriptAccumulator }) {
    return (
      <div className="space-y-4">
        {Object.entries(transcripts).map(([responseId, { text, timestamp }]) => (
          <Card key={responseId} className="w-full bg-white shadow-sm">
            <div className="p-6">
              <div className="flex items-start space-x-4">
                <div className="h-12 w-12 rounded-full bg-blue-100 flex items-center justify-center">
                  <MessageSquare className="h-6 w-6 text-blue-600" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-500">
                        {new Date(timestamp).toLocaleTimeString()}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        ID: {responseId.slice(0, 8)}
                      </Badge>
                    </div>
                  </div>
                  <p className="mt-2 text-gray-800">{text}</p>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    );
  }


  export default TranscriptView;