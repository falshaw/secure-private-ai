// Tests for the built package (npm test builds first).
import { test } from "node:test";
import assert from "node:assert/strict";
import { SignJWT } from "jose";
import {
  BCRYPT_COST,
  hashPassword,
  verifyPassword,
  encodeJwtSecret,
  signHs256,
  verifyHs256,
} from "../dist/auth-core.js";

const SECRET = "a-test-secret-that-is-long-enough-0123456789";

test("bcrypt hashes at cost 12 and verifies only the right password", async () => {
  const hash = await hashPassword("correct horse battery staple");
  assert.equal(BCRYPT_COST, 12);
  assert.match(hash, /^\$2[aby]\$12\$/);
  assert.equal(await verifyPassword("correct horse battery staple", hash), true);
  assert.equal(await verifyPassword("wrong password", hash), false);
});

test("encodeJwtSecret refuses a missing or short secret", () => {
  assert.throws(() => encodeJwtSecret(undefined), /required/);
  assert.throws(() => encodeJwtSecret("short", { minLength: 32 }), /at least 32/);
  assert.equal(encodeJwtSecret(SECRET, { minLength: 32 }).length, SECRET.length);
});

test("a signed token verifies and keeps its claims and jti", async () => {
  const key = encodeJwtSecret(SECRET);
  const token = await signHs256({ sub: "u1", role: "admin" }, key, { expiresIn: "1h", jti: "id-1" });
  const payload = await verifyHs256(token, key);
  assert.equal(payload.sub, "u1");
  assert.equal(payload.role, "admin");
  assert.equal(payload.jti, "id-1");
  assert.ok(payload.exp - payload.iat === 3600);
});

test("verification fails for another key, an expired token and a tampered token", async () => {
  const key = encodeJwtSecret(SECRET);
  const token = await signHs256({ sub: "u1" }, key);
  await assert.rejects(verifyHs256(token, encodeJwtSecret(SECRET + "x")));

  const expired = await signHs256({ sub: "u1" }, key, { expiresIn: Math.floor(Date.now() / 1000) - 60 });
  await assert.rejects(verifyHs256(expired, key), /exp/);

  const [h, , s] = token.split(".");
  const forged = `${h}.${Buffer.from(JSON.stringify({ sub: "admin" })).toString("base64url")}.${s}`;
  await assert.rejects(verifyHs256(forged, key));
});

test("the algorithm is pinned: an HS512 token with the same key is refused", async () => {
  const key = encodeJwtSecret(SECRET);
  const hs512 = await new SignJWT({ sub: "u1" }).setProtectedHeader({ alg: "HS512" }).setExpirationTime("1h").sign(key);
  await assert.rejects(verifyHs256(hs512, key), /alg/);
});
