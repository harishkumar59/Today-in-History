import { NextResponse } from 'next/server';

export const runtime = "nodejs";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const API_BASE = 'https://generativelanguage.googleapis.com/v1beta';

// Fallback model chain — each has a separate free-tier quota
// If one hits 429 the next one is tried automatically
const MODEL_CHAIN = [
  process.env.GEMINI_MODEL || 'gemini-1.5-flash',
  'gemini-1.5-flash-8b',
  'gemini-1.0-pro',
];

// Define message type
type Message = {
  role: string;
  content: string;
};

/**
 * Try calling a single Gemini model. Returns the Response object so the
 * caller can inspect the status code and decide whether to fall through.
 */
async function callModel(model: string, body: object, signal: AbortSignal): Promise<Response> {
  const url = `${API_BASE}/models/${model}:generateContent?key=${GEMINI_API_KEY}`;
  console.log(`Trying model: ${model}`);
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
}

export async function POST(req: Request) {
  try {
    // ── Guard: API key ─────────────────────────────────────────────────────────
    if (!GEMINI_API_KEY) {
      console.error('Missing Gemini API key in environment variables');
      return NextResponse.json(
        { error: 'API key is not configured. Please add GEMINI_API_KEY to your environment variables.' },
        { status: 500 }
      );
    }

    const { messages }: { messages: Message[] } = await req.json();
    const dateQuery = messages[messages.length - 1].content;

    // ── Build prompt ───────────────────────────────────────────────────────────
    const historyPrompt = `
You are a history expert AI assistant. A user is asking about historical events that happened on a specific date.

Query: "${dateQuery}"

Respond with 3-5 significant historical events that occurred on this date throughout history. For each event:
1. Include the year
2. Provide a brief, engaging description of the event
3. Focus on diverse events from different time periods and categories (politics, science, arts, etc.)
4. Present the information in a clear, engaging format
5. Add interesting details that make the history come alive

Make your response engaging, educational, and well-formatted.
`;

    const requestBody = {
      contents: [{ parts: [{ text: historyPrompt }] }],
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 4096,
      },
    };

    // ── Model fallback loop ────────────────────────────────────────────────────
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000); // 20 s total

    let lastStatus = 0;
    let lastErrorText = '';

    try {
      for (const model of MODEL_CHAIN) {
        let response: Response;

        try {
          response = await callModel(model, requestBody, controller.signal);
        } catch (fetchErr) {
          if (fetchErr instanceof Error && fetchErr.name === 'AbortError') {
            clearTimeout(timeoutId);
            return NextResponse.json(
              { error: 'The request timed out. Please try again.' },
              { status: 504 }
            );
          }
          // Network error — skip to next model
          console.warn(`Network error with model ${model}:`, fetchErr);
          continue;
        }

        // ── Success ────────────────────────────────────────────────────────────
        if (response.ok) {
          clearTimeout(timeoutId);
          const data = await response.json();

          let text = '';
          if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
            text = data.candidates[0].content.parts[0].text;
          }
          if (text.startsWith('Assistant:')) {
            text = text.substring('Assistant:'.length).trim();
          }
          if (!text) {
            text = "Sorry, I couldn't find any significant historical events for that date.";
          }

          console.log(`Response from ${model}:`, text.substring(0, 100) + '...');
          return NextResponse.json({ text });
        }

        // ── Rate-limited (429) — try next model ────────────────────────────────
        if (response.status === 429) {
          lastStatus = 429;
          lastErrorText = await response.text();
          console.warn(`Model ${model} is rate-limited (429). Trying next model...`);
          continue;
        }

        // ── Other HTTP error — surface immediately ──────────────────────────────
        lastStatus = response.status;
        lastErrorText = await response.text();
        console.error(`Model ${model} returned ${response.status}:`, lastErrorText);

        try {
          const errorData = JSON.parse(lastErrorText);
          clearTimeout(timeoutId);
          return NextResponse.json(
            { error: `Gemini API error: ${response.status}`, details: errorData },
            { status: response.status }
          );
        } catch {
          clearTimeout(timeoutId);
          return NextResponse.json(
            { error: `Gemini API error: ${response.status}`, details: lastErrorText },
            { status: response.status }
          );
        }
      }

      // ── All models exhausted (all hit 429) ─────────────────────────────────
      clearTimeout(timeoutId);
      console.error('All Gemini models are rate-limited.');
      return NextResponse.json(
        {
          error:
            'The Gemini API is temporarily rate-limited. Please wait a minute and try again.',
          details: lastErrorText,
        },
        { status: 429 }
      );

    } catch (outerErr) {
      clearTimeout(timeoutId);
      if (outerErr instanceof Error && outerErr.name === 'AbortError') {
        return NextResponse.json(
          { error: 'The request timed out. Please try again.' },
          { status: 504 }
        );
      }
      throw outerErr;
    }

  } catch (error) {
    console.error('Error in Gemini API route:', error);
    return NextResponse.json(
      { error: 'An unexpected error occurred. Please try again.' },
      { status: 500 }
    );
  }
}