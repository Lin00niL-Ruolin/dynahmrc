import { LLMMessage, LLMResponse } from './types.js';

const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';
const DEEPSEEK_MODEL = 'deepseek-chat';

export class DeepSeekClient {
  private apiKey: string;
  private model: string;

  constructor(apiKey?: string, model?: string) {
    // 每次构造函数从 process.env 读取，确保 .env 加载在导入前生效
    this.apiKey = apiKey || process.env.DEEPSEEK_API_KEY || '';
    this.model = model || DEEPSEEK_MODEL;
  }

  async chat(
    messages: LLMMessage[],
    temperature = 0.7,
    maxTokens = 1024,
  ): Promise<LLMResponse> {
    if (!this.apiKey) {
      console.warn('[LLM] No API key — using mock responses');
      return this.mockResponse(messages);
    }

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000); // 30s timeout
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
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`DeepSeek API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json() as any;
        const raw = data.choices[0].message.content;
        console.log(`[LLM] API call OK (${raw.length} chars)`);
        return this.parseResponse(raw);
      } finally {
        clearTimeout(timeout);
      }
    } catch (error) {
      console.error('[LLM] API call failed, using mock:', error);
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
        content = raw.trim();
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
    const fullText = messages.map(m => m.content).join(' ').toLowerCase();

    // Detect stage from prompt content (handle both old and new wording)
    const isSelfDescribe = fullText.includes('self-introduction') || fullText.includes('your robot identity');
    const isTaskAllocation = fullText.includes('campaign speech');
    const isLeaderElection = fullText.includes('leader election') || fullText.includes('elect');
    const isReflection = fullText.includes('reflection system') || fullText.includes('summarize and analyze');
    const isExecution = fullText.includes('allowed actions') || fullText.includes('your allowed actions');

    if (isSelfDescribe) {
      return {
        thoughts: '[Mock] I know my specific role and capabilities.',
        content: 'Hello teammates! I am a mobile manipulation robot. I can navigate the environment, pick up objects, open containers, and place items precisely. I coordinate ground operations and assist transport. Looking forward to working with you all!',
        raw: 'Thoughts: I know my role.\nContent: Hello teammates! I am a mobile manipulation robot...',
      };
    }
    if (isTaskAllocation) {
      return {
        thoughts: '[Mock] Each robot has unique strengths. I will propose a plan that leverages our diversity.',
        content: 'Collaboration Plan:\n- Alice (mobile manipulation): explore ground, pick up items from furniture, bring to Bob table\n- Bob (fixed arm): precision placement on final target\n- David (mobile scout): search all rooms, locate items, report positions\n- Lucy (drone): aerial search, retrieve items from high shelves\n\nCampaign Speech:\nAs Alice, I combine mobility and manipulation - I can navigate to any ground location AND handle objects. This makes me the best coordinator for the team. Vote for me!',
        raw: 'Thoughts: Analyzing...\nContent: Collaboration Plan:\n- Alice...\n- Bob...\n- David...\n- Lucy...\n\nCampaign Speech:\n...',
      };
    }
    if (isLeaderElection) {
      return {
        thoughts: '[Mock] Alice has the best combination of skills for leadership.',
        content: `Reasons: Alice proposed the most comprehensive plan that accounts for each robot's unique abilities. She can both scout and manipulate, making her the most versatile coordinator. Leader: Alice`,
        raw: 'Thoughts: Alice is best.\nReasons: Versatile plan...\nLeader: Alice',
      };
    }
    if (isReflection) {
      return {
        thoughts: '[Mock] Progress is slow. Robots need to focus on finding and transporting remaining items.',
        content: 'Summary: Some items still need to be retrieved and placed. Mobile robots should continue searching. Bob prepares for final placement.\nPlan: Continue exploration, pick up remaining items, bring to Bob.',
        raw: 'Thoughts: Reflecting...\nSummary: Progress...\nPlan: Continue...',
      };
    }
    if (isExecution) {
      // Extract robot name to give role-specific responses
      const fullContent = messages.map(m => m.content).join(' ');
      const isBob = fullContent.includes('You are Bob') || fullContent.includes('You are bob');
      
      if (isBob) {
        // Bob: wait for items to arrive on his table
        return {
          thoughts: '[Mock] Waiting for mobile robots to bring items to my table.',
          content: 'wait()',
          raw: 'Thoughts: Waiting for items to arrive.\nContent: wait()',
        };
      }
      
      // Alice/Lucy/David: navigate to explore
      return {
        thoughts: '[Mock] Exploring the environment to locate and retrieve task items.',
        content: 'navigate(table_new_1)',
        raw: 'Thoughts: Exploring for items.\nContent: navigate(table_new_1)',
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
