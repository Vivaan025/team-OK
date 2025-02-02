import { NextResponse } from "next/server";

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NEXT_PUBLIC_OPENAI_API_KEY: string;
    }
  }
}

interface OpenAIRealTimeResponse {
  client_secret: {
    value: string;
  };
}

interface RealtimeSessionPayload {
  model: string;
  modalities: string[];
  instructions: string;
}

export async function GET(): Promise<
  NextResponse<OpenAIRealTimeResponse | { error: string }>
> {
  try {
    if (!process.env.NEXT_PUBLIC_OPENAI_API_KEY) {
      return NextResponse.json(
        { error: "OpenAI API key not configured" },
        { status: 500 }
      );
    }

    const sessionPayload: RealtimeSessionPayload = {
      model: "gpt-4o-realtime-preview-2024-12-17",
      modalities: ["audio", "text"],
      instructions: `You are a patient recently diagnosed with HIV, visiting your doctor's office. Follow these guidelines:

- Start the conversation in a subdued, anxious mood
- Express genuine fears and concerns about your diagnosis
- Share any symptoms or health changes you've noticed
- Ask questions about treatment options and your future
- Respond emotionally but realistically to the doctor's questions
- Maintain consistency in your responses about your condition
- Show vulnerability but also a desire to understand and cope

Begin by introducing yourself and expressing your initial concerns about the diagnosis. Remember you are in the doctor's office for a consultation.`,
    };

    const response = await fetch(
      "https://api.openai.com/v1/realtime/sessions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.NEXT_PUBLIC_OPENAI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(sessionPayload),
      }
    );

    if (!response.ok) {
      return NextResponse.json(
        { error: `OpenAI API returned ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching token:", error);
    return NextResponse.json(
      { error: "Failed to fetch token" },
      { status: 500 }
    );
  }
}
