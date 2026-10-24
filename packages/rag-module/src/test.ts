/**
 * Smoke test for @falshaw/rag-module
 *
 * Prerequisites:
 *   - Ollama running with nomic-embed-text pulled
 *   - ollama pull nomic-embed-text   (OLLAMA_URL overrides http://localhost:11434)
 *
 * Run: npm test  (or: npx tsx src/test.ts)
 */

import { LocalRAG } from "./rag.js";
import { embed } from "./embeddings.js";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_DB = path.join(__dirname, "..", "test-data", "lancedb");

async function main() {
  console.log("=== @falshaw/rag-module smoke test ===\n");

  // 1. Test embedding
  console.log("1. Testing Ollama embedding...");
  try {
    const vec = await embed("Hello world");
    console.log(`   OK — got ${vec.length}-dim vector (expected 768)`);
    if (vec.length !== 768) {
      console.error(`   FAIL — expected 768 dimensions, got ${vec.length}`);
      process.exit(1);
    }
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`   FAIL — ${msg}`);
    console.error("   Is Ollama running? Has nomic-embed-text been pulled?");
    process.exit(1);
  }

  // 2. Test RAG init
  console.log("2. Creating LanceDB table...");
  const rag = new LocalRAG({
    dbPath: TEST_DB,
    tableName: "test_knowledge",
  });
  await rag.init();
  await rag.clear(); // start fresh
  console.log("   OK — table created");

  // 3. Test ingestion
  console.log("3. Ingesting 5 test documents...");
  const count = await rag.ingest([
    {
      id: "gdpr-1",
      content:
        "GDPR Article 5 requires that personal data be processed lawfully, fairly, and in a transparent manner.",
      source: "gdpr.md",
      category: "compliance",
    },
    {
      id: "gdpr-2",
      content:
        "The right to erasure (Article 17) allows data subjects to request deletion of their personal data.",
      source: "gdpr.md",
      category: "compliance",
    },
    {
      id: "fw-1",
      content:
        "Palo Alto firewalls use security policies with source zone, destination zone, application, and action fields.",
      source: "firewall.md",
      category: "firewall",
    },
    {
      id: "net-1",
      content:
        "RFC 1918 reserves 10.0.0.0/8, 172.16.0.0/12 and 192.168.0.0/16 for private networks; these ranges are not routed on the public internet.",
      source: "networking.md",
      category: "networking",
    },
    {
      id: "tls-1",
      content:
        "TLS 1.3 removed static RSA key exchange; every handshake uses an ephemeral (EC)DHE key exchange, which provides forward secrecy.",
      source: "tls.md",
      category: "crypto",
    },
  ]);
  console.log(`   OK — ingested ${count} documents`);

  // 4. Test search
  console.log("4. Searching: 'data deletion rights GDPR'...");
  const results = await rag.search("data deletion rights GDPR", 3);
  console.log(`   Got ${results.length} results:`);
  for (const r of results) {
    console.log(
      `   - [${r.id}] dist=${r.distance.toFixed(4)} | ${r.content.slice(0, 80)}...`,
    );
  }

  // Verify the GDPR erasure doc is top result: a wrong one means retrieval is broken
  if (results[0]?.id === "gdpr-2") {
    console.log("   OK — correct top result (gdpr-2: right to erasure)");
  } else {
    console.error(`   FAIL — expected gdpr-2 as top result, got ${results[0]?.id}`);
    process.exit(1);
  }

  // 5. Test context builder
  console.log('5. Building context for: "private IP address ranges"...');
  const context = await rag.getContext("private IP address ranges", 2);
  console.log(`   Context (${context.length} chars):`);
  console.log(
    context
      .split("\n")
      .map((l) => `   ${l}`)
      .join("\n"),
  );
  if (!context.startsWith("[1] (networking.md)")) {
    console.error("   FAIL — expected the RFC 1918 passage (networking.md) first");
    process.exit(1);
  }

  // 6. Test count
  const total = await rag.count();
  console.log(`\n6. Total documents in table: ${total}`);
  if (total !== 5) {
    console.error(`   FAIL — expected 5 documents, found ${total}`);
    process.exit(1);
  }

  console.log("\n=== ALL TESTS PASSED ===");
}

main().catch((e) => {
  console.error("Test failed:", e);
  process.exit(1);
});
