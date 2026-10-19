/**
 * @falshaw/auth — shared auth core.
 *
 * The security-critical primitives that every custom-JWT app needs and that must
 * stay identical across apps: bcrypt password hashing and HS256 sign/verify with
 * the algorithm pinned. App-specific concerns (payload shape, jti policy,
 * null-vs-throw on verify, cookie name, session storage) stay in each app's own
 * auth module.
 *
 * Use it as a normal dependency, or vendor (copy) this file into apps that build
 * in an isolated Docker context, where a package outside the app directory does
 * not exist at build time. When vendoring, edit only this canonical copy and
 * re-copy it into each app.
 */
import { SignJWT, jwtVerify, type JWTPayload } from "jose";
import bcrypt from "bcryptjs";

/** bcrypt work factor — consistent across all apps. */
export const BCRYPT_COST = 12;

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, BCRYPT_COST);
}

export async function verifyPassword(
  password: string,
  hash: string,
): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

/**
 * Encode JWT_SECRET into an HS256 key. Pass `minLength` to enforce a minimum
 * secret length (apps that previously required >=32 chars should pass 32).
 */
export function encodeJwtSecret(
  secret: string | undefined,
  opts: { minLength?: number } = {},
): Uint8Array {
  if (!secret) {
    throw new Error("JWT_SECRET environment variable is required");
  }
  if (opts.minLength && secret.length < opts.minLength) {
    throw new Error(
      `JWT_SECRET must be at least ${opts.minLength} characters for HS256 security`,
    );
  }
  return new TextEncoder().encode(secret);
}

/** Sign an HS256 JWT. Pass `jti` to include a unique token id. */
export async function signHs256(
  payload: JWTPayload,
  secret: Uint8Array,
  opts: { expiresIn?: string | number; jti?: string } = {},
): Promise<string> {
  let builder = new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(opts.expiresIn ?? "24h");
  if (opts.jti) builder = builder.setJti(opts.jti);
  return builder.sign(secret);
}

/**
 * Verify an HS256 JWT with the algorithm pinned (prevents alg-substitution).
 * Throws on invalid/expired token — callers wrap this to return null if they
 * prefer a non-throwing contract.
 */
export async function verifyHs256(
  token: string,
  secret: Uint8Array,
): Promise<JWTPayload> {
  const { payload } = await jwtVerify(token, secret, {
    algorithms: ["HS256"],
  });
  return payload;
}
