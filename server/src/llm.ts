import { LLMMessage, LLMResponse } from './types.js';

const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || '';
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';
const DEEPSEEK_MODEL = 'deepseek-chat';

export class DeepSeekClient {
  private apiKey: string;
  private model: string;

  constructor(apiKey?: string, model?: string) {
    this.apiKey = apiKey || DEEPSEEK_API_KEY;
    this.model = model || DEEPSEEK_MODEL;
  }

  async chat(
    messages: LLMMessage[],
    temperature = 0.7,
    maxTokens = 2048,
  ): Promise<LLMResponse> {
    if (!this.apiKey) {
      return this.mockResponse(messages);
    }

    try {
      const response = await fetch(DEEPSEEK_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          messages,
          temperature,
          max_tokens: maxTokens,
        }),
      });

      if (!response.ok) {
        throw new Error(`DeepSeek API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json() as any;
      const raw = data.choices[0].message.content;
      return this.parseResponse(raw);
    } catch (error) {
      console.error('[ERROR] DeepSeek API call failed:', error);
      return this.mockResponse(messages);
    }
  }

  private parseResponse(raw: string): LLMResponse {
    let thoughts = '';
    let content = '';
    let inThoughts = false;
    let inContents = false;

    try {
      const lines = raw.split('\n');
      for (const line of lines) {
        const stripped = line.trim().toLowerCase();

        if (stripped.startsWith('thoughts') || stripped.startsWith('thought')) {
          inThoughts = true;
          inContents = false;
          const colonIdx = line.indexOf(':');
          if (colonIdx >= 0) {
            thoughts += line.substring(colonIdx + 1).trim() + '\n';
          }
          continue;
        }

        if (stripped.startsWith('contents') || stripped.startsWith('content') ||
            stripped.startsWith('reasons') || stripped.startsWith('leader') ||
            stripped.startsWith('summaries') || stripped.startsWith('plans')) {
          inContents = true;
          inThoughts = false;
          content += line + '\n';
          continue;
        }

        if (inThoughts) thoughts += line + '\n';
        else if (inContents) content += line + '\n';
      }

      if (!thoughts && !content) {
        content = raw;
      }
    } catch {
      content = raw;
    }

    return {
      thoughts: thoughts.trim(),
      content: content.trim(),
      raw,
    };
  }

  private mockResponse(_messages: LLMMessage[]): LLMResponse {
    return {
      thoughts: '[Mock] Analyzing the task and my capabilities...',
      content: '[Mock] I will navigate to the table and pick up the object.',
      raw: '[Mock Response]\nThoughts: Analyzing...\nContent: Executing action...',
    };
  }
}

let _client: DeepSeekClient | null = null;

export function getLLMClient(): DeepSeekClient {
  if (!_client) {
    _client = new DeepSeekClient();
  }
  return _client;
}
