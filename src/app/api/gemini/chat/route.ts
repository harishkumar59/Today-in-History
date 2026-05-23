import { NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

export const runtime = 'nodejs';

// Modern Model fallback chain - updated to active production models
const MODEL_CHAIN = [
  'gemini-2.5-flash',
  'gemini-2.0-flash',
  'gemini-1.5-flash'
];

type Message = {
  role: string;
  content: string;
};

export async function POST(req: Request) {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return NextResponse.json(
      { error: 'GEMINI_API_KEY is not set in environment variables.' },
      { status: 500 }
    );
  }

  let messages: Message[];
  try {
    ({ messages } = await req.json());
  } catch {
    return NextResponse.json({ error: 'Invalid request body.' }, { status: 400 });
  }

  const dateQuery = messages[messages.length - 1]?.content ?? '';

  const prompt = `You are a history expert AI assistant. A user is asking about historical events that happened on a specific date.

Query: "${dateQuery}"

Respond with 3-5 significant historical events that occurred on this date throughout history. For each event:
1. Include the year
2. Provide a brief, engaging description of the event
3. Focus on diverse events from different time periods and categories (politics, science, arts, etc.)
4. Present the information in a clear, engaging format
5. Add interesting details that make the history come alive

Make your response engaging, educational, and well-formatted.`;

  const genAI = new GoogleGenerativeAI(apiKey);

  for (const modelName of MODEL_CHAIN) {
    try {
      console.log(`Trying model: ${modelName}`);
      const model = genAI.getGenerativeModel({ model: modelName });

      const result = await model.generateContent(prompt);
      const text = result.response.text();

      console.log(`Success with model: ${modelName}`);
      return NextResponse.json({ text });

    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      console.warn(`Model ${modelName} failed: ${message}`);

      // Catching any routing or quota failures to slip gracefully to the next tier
      if (
        message.includes('429') ||
        message.includes('404') ||
        message.toLowerCase().includes('quota') ||
        message.toLowerCase().includes('not found') ||
        message.toLowerCase().includes('unsupported')
      ) {
        console.warn(`Skipping model ${modelName}, trying next...`);
        continue;
      }

      // Any other structural code error → surface it immediately
      return NextResponse.json(
        { error: `Gemini API error: ${message}` },
        { status: 500 }
      );
    }
  }

  // All models exhausted
  return NextResponse.json(
    { error: 'All current Gemini models are hitting tier limits. Please wait a moment and try again.' },
    { status: 429 }
  );
}