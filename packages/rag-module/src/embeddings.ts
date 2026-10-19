/**
 * Ollama embedding client — nomic-embed-text by default.
 * The Ollama URL defaults to OLLAMA_URL, then http://localhost:11434.
 */

export interface EmbeddingOptions {
  ollamaUrl?: string;
  model?: string;
}

const DEFAULTS = {
  ollamaUrl: process.env.OLLAMA_URL ?? "http://localhost:11434",
  model: "nomic-embed-text",
} as const;

/**
 * Get embedding vector for a single text string.
 */
export async function embed(
  text: string,
  options?: EmbeddingOptions,
): Promise<number[]> {
  const url = options?.ollamaUrl ?? DEFAULTS.ollamaUrl;
  const model = options?.model ?? DEFAULTS.model;

  const response = await fetch(`${url}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input: text }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Ollama embed failed (${response.status}): ${body}`);
  }

  const data = (await response.json()) as { embeddings: number[][] };
  return data.embeddings[0];
}

/**
 * Get embedding vectors for multiple texts in one call.
 */
export async function embedBatch(
  texts: string[],
  options?: EmbeddingOptions,
): Promise<number[][]> {
  const url = options?.ollamaUrl ?? DEFAULTS.ollamaUrl;
  const model = options?.model ?? DEFAULTS.model;

  const response = await fetch(`${url}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input: texts }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Ollama embed batch failed (${response.status}): ${body}`);
  }

  const data = (await response.json()) as { embeddings: number[][] };
  return data.embeddings;
}
