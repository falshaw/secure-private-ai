/**
 * RAG module — LanceDB-backed retrieval-augmented generation.
 *
 * Each app creates its own RAG instance with a domain-specific table.
 * Knowledge is stored on disk (LanceDB is embedded, no server needed).
 */

import * as lancedb from "@lancedb/lancedb";
import { embed, embedBatch, type EmbeddingOptions } from "./embeddings.js";

// nomic-embed-text produces 768-dimensional vectors
const VECTOR_DIM = 768;

export interface RagConfig {
  /** Path to LanceDB storage directory (e.g. "./data/lancedb") */
  dbPath: string;
  /** Table name for this domain (e.g. "network_security", "compliance_de") */
  tableName: string;
  /** Ollama connection options */
  embedding?: EmbeddingOptions;
}

export interface Document {
  /** Unique identifier */
  id: string;
  /** The text content to embed and search */
  content: string;
  /** Source file or URL this came from */
  source: string;
  /** Optional category/tag */
  category?: string;
}

interface StoredDocument {
  id: string;
  content: string;
  source: string;
  category: string;
  vector: number[];
}

export interface SearchResult {
  id: string;
  content: string;
  source: string;
  category: string;
  distance: number;
}

export class LocalRAG {
  private config: RagConfig;
  private db: lancedb.Connection | null = null;
  private table: lancedb.Table | null = null;

  constructor(config: RagConfig) {
    this.config = config;
  }

  /**
   * Connect to LanceDB and open (or create) the table.
   */
  async init(): Promise<void> {
    this.db = await lancedb.connect(this.config.dbPath);

    const tableNames = await this.db.tableNames();
    if (tableNames.includes(this.config.tableName)) {
      this.table = await this.db.openTable(this.config.tableName);
    } else {
      // Create with a dummy row to establish schema, then delete it
      const dummyVector = new Array(VECTOR_DIM).fill(0);
      this.table = await this.db.createTable(this.config.tableName, [
        {
          id: "__init__",
          content: "",
          source: "",
          category: "",
          vector: dummyVector,
        },
      ]);
      await this.table.delete('id = "__init__"');
    }
  }

  private ensureTable(): lancedb.Table {
    if (!this.table) throw new Error("Call init() before using LocalRAG");
    return this.table;
  }

  /**
   * Add documents to the knowledge base. Embeds them automatically.
   */
  async ingest(docs: Document[]): Promise<number> {
    const table = this.ensureTable();

    // Embed in batches of 32
    const batchSize = 32;
    const records: StoredDocument[] = [];

    for (let i = 0; i < docs.length; i += batchSize) {
      const batch = docs.slice(i, i + batchSize);
      const texts = batch.map((d) => d.content);
      const vectors = await embedBatch(texts, this.config.embedding);

      for (let j = 0; j < batch.length; j++) {
        records.push({
          id: batch[j].id,
          content: batch[j].content,
          source: batch[j].source,
          category: batch[j].category ?? "",
          vector: vectors[j],
        });
      }
    }

    await table.add(records);
    return records.length;
  }

  /**
   * Search for relevant documents given a query string.
   */
  async search(
    query: string,
    limit: number = 5,
    filter?: string,
  ): Promise<SearchResult[]> {
    const table = this.ensureTable();
    const queryVector = await embed(query, this.config.embedding);

    let q = table.search(queryVector).limit(limit);
    if (filter) {
      q = q.where(filter);
    }

    const results = await q.toArray();

    return results.map((r: Record<string, unknown>) => ({
      id: r["id"] as string,
      content: r["content"] as string,
      source: r["source"] as string,
      category: r["category"] as string,
      distance: r["_distance"] as number,
    }));
  }

  /**
   * Build a context string from search results, ready to inject into a prompt.
   */
  async getContext(
    query: string,
    limit: number = 5,
    filter?: string,
  ): Promise<string> {
    const results = await this.search(query, limit, filter);
    if (results.length === 0) return "";

    return results
      .map((r, i) => `[${i + 1}] (${r.source})\n${r.content}`)
      .join("\n\n");
  }

  /**
   * Get row count in the table.
   */
  async count(): Promise<number> {
    const table = this.ensureTable();
    return table.countRows();
  }

  /**
   * Delete all documents (drop and recreate table).
   */
  async clear(): Promise<void> {
    if (!this.db) throw new Error("Call init() before using LocalRAG");
    await this.db.dropTable(this.config.tableName);
    this.table = null;
    await this.init();
  }
}
