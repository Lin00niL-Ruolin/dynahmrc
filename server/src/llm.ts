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

  private mockResponse(messages: LLMMessage[]): LLMResponse {
    // Try to produce a valid action if this is an execution-stage call
    const isExecution = messages.some(m => m.content.includes('AVAILABLE ACTIONS'));
    const isSelfDescribe = messages.some(m => m.content.includes('self-introduction'));
    const isTaskAllocation = messages.some(m => m.content.includes('campaign speech'));
    const isLeaderElection = messages.some(m => m.content.includes('Leader Election'));
    const isReflection = messages.some(m => m.content.includes('Reflection') || m.content.includes('reflection'));

    if (isSelfDescribe) {
      return {
        thoughts: '[Mock] I am a capable robot. I can navigate, pick, and place objects to help complete the mission.',
        content: 'Hello teammates! I am a versatile mobile robot. I can efficiently navigate the environment, pick up objects, and place them at targets. I will assist in locating and transporting items to complete our shared mission.',
        raw: 'Thoughts: I am...\nContent: Hello teammates!...',
      };
    }
    if (isTaskAllocation) {
      return {
        thoughts: '[Mock] Analyzing the team composition and task requirements to propose an effective plan.',
        content: 'Collaboration Plan:\n- Alice will explore and collect task items from the environment\n- Bob will handle precise placement at the target location\n- David will assist in navigation and communication\n- Lucy will scout from above and provide aerial support\n\nCampaign Speech:\nI have the best combination of mobility and manipulation for this task. I propose an efficient parallel workflow where each robot works simultaneously on different subtasks. Vote for me as leader!',
        raw: 'Thoughts: Analyzing...\nContent: Collaboration Plan:...',
      };
    }
    if (isLeaderElection) {
      return {
        thoughts: '[Mock] Reviewing all plans and speeches to select the best leader.',
        content: `After careful analysis, Alice has the most comprehensive plan that leverages all robots\' strengths effectively. Reasons: Alice proposed clear parallel task assignments and demonstrated strong coordination skills. Leader: Alice`,
        raw: 'Thoughts: Reviewing...\nContent: Reasons:...\nLeader: Alice',
      };
    }
    if (isReflection) {
      return {
        thoughts: '[Mock] Reflecting on the progress so far and planning next steps.',
        content: 'Summary: We have made progress on collecting items. Some remain to be found and placed. Plan: Continue exploring, pick up remaining items, and deliver them to the target location.',
        raw: 'Thoughts: Reflecting...\nSummary: Progress...\nPlan: Continue...',
      };
    }
    if (isExecution) {
      // Pick a random furniture name from the scene graph in messages
      const furnMatch = messages[messages.length-1]?.content.match(/navigate\(\w+\)/);
      const target = furnMatch ? furnMatch[0].replace('navigate(','').replace(')','') : 'table_0';
      return {
        thoughts: `[Mock] Moving to ${target} to search for task items.`,
        content: `navigate(${target})`,
        raw: `Thoughts: Moving to ${target}...\nContent: navigate(${target})`,
      };
    }

    return {
      thoughts: '[Mock] Ready to proceed.',
      content: 'wait()',
      raw: '[Mock]\nThoughts: Ready.\nContent: wait()',
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
