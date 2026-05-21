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
        // 去掉行首编号，如 "1) Thoughts:" → "Thoughts:", "2. Contents:" → "Contents:"
        const clean = line.trim().replace(/^[\d\.\]\)\s]+/, '').toLowerCase();
        const original = line.trim();

        if (clean.startsWith('thoughts') || clean.startsWith('thought')) {
          inThoughts = true;
          inContents = false;
          // 找第一个冒号截取
          const colonIdx = original.indexOf(':');
          if (colonIdx >= 0) {
            thoughts += original.substring(colonIdx + 1).trim() + '\n';
          } else {
            inThoughts = true;
          }
          continue;
        }

        if (clean.startsWith('contents') || clean.startsWith('content') ||
            clean.startsWith('reasons') || clean.startsWith('leader') ||
            clean.startsWith('summaries') || clean.startsWith('plans') ||
            clean.startsWith('summary') || clean.startsWith('plan')) {
          inContents = true;
          inThoughts = false;
          content += original + '\n';
          continue;
        }

        if (inThoughts) thoughts += original + '\n';
        else if (inContents) content += original + '\n';
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
