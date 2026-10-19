/**
 * Knowledge ingestion — loads JSON knowledge files into LanceDB tables.
 *
 * Each file in knowledge/ is ingested into its own table, named after the file
 * (dashes become underscores: network-security.json -> network_security).
 *
 * Usage:
 *   npx tsx src/ingest.ts                         # ingest every knowledge/*.json
 *   npx tsx src/ingest.ts network-security        # ingest knowledge/network-security.json only
 *   npx tsx src/ingest.ts ./my-docs.json          # ingest any file with the same shape
 *
 * Environment:
 *   RAG_DB_PATH   LanceDB directory (default: ./data/lancedb inside this package)
 *   OLLAMA_URL    Ollama server for embeddings (default: http://localhost:11434)
 *
 * File format: [{ "id": "...", "source": "...", "category": "...", "content": "..." }]
 *
 * Prerequisites:
 *   - Ollama running with nomic-embed-text pulled
 */

import { LocalRAG, type Document } from "./rag.js";
import { existsSync, readFileSync, readdirSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const KNOWLEDGE_DIR = path.join(__dirname, "..", "knowledge");
const DB_PATH =
  process.env.RAG_DB_PATH ?? path.join(__dirname, "..", "data", "lancedb");

interface KnowledgeEntry {
  id: string;
  source: string;
  category: string;
  content: string;
}

function resolveFile(arg: string): string {
  if (arg.endsWith(".json") && existsSync(arg)) return path.resolve(arg);
  const inKnowledge = path.join(
    KNOWLEDGE_DIR,
    arg.endsWith(".json") ? arg : `${arg}.json`,
  );
  if (existsSync(inKnowledge)) return inKnowledge;
  throw new Error(`Knowledge file not found: ${arg}`);
}

function tableNameFor(filePath: string): string {
  return path
    .basename(filePath, ".json")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_");
}

async function ingestFile(filePath: string): Promise<void> {
  const tableName = tableNameFor(filePath);
  console.log(`\n--- ${tableName} ---`);
  console.log(`  Knowledge file: ${filePath}`);
  console.log(`  LanceDB path:   ${DB_PATH}`);

  const entries: KnowledgeEntry[] = JSON.parse(readFileSync(filePath, "utf-8"));
  console.log(`  Documents:      ${entries.length}`);

  const docs: Document[] = entries.map((e) => ({
    id: e.id,
    content: e.content,
    source: e.source,
    category: e.category,
  }));

  const rag = new LocalRAG({ dbPath: DB_PATH, tableName });
  await rag.init();
  await rag.clear(); // fresh ingest each time
  const count = await rag.ingest(docs);
  console.log(`  Ingested:       ${count} documents`);

  // Sanity check: the first document's own text should retrieve itself.
  const probe = docs[0]?.content.slice(0, 200);
  if (probe) {
    const results = await rag.search(probe, 1);
    console.log(
      `  Self-check:     [${results[0]?.id}] dist=${results[0]?.distance.toFixed(4)} (expected ${docs[0].id})`,
    );
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const files =
    args.length > 0
      ? args.map(resolveFile)
      : readdirSync(KNOWLEDGE_DIR)
          .filter((f) => f.endsWith(".json"))
          .map((f) => path.join(KNOWLEDGE_DIR, f));

  console.log("=== Knowledge ingestion ===");
  for (const f of files) {
    await ingestFile(f);
  }
  console.log("\n=== INGESTION COMPLETE ===");
}

main().catch((e) => {
  console.error("Ingestion failed:", e);
  process.exit(1);
});
