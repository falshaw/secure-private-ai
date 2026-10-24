# @falshaw/auth — shared auth core

`auth-core.ts` holds the security-critical auth primitives that several apps
share: bcrypt password hashing and HS256 JWT signing/verification via
[`jose`](https://github.com/panva/jose), with the algorithm pinned.

## API

| Export | Purpose |
| --- | --- |
| `BCRYPT_COST` | bcrypt work factor (12) |
| `hashPassword(password)` / `verifyPassword(password, hash)` | bcrypt hash / compare |
| `encodeJwtSecret(secret, { minLength? })` | Turns `JWT_SECRET` into an HS256 key; throws if missing or shorter than `minLength` |
| `signHs256(payload, key, { expiresIn?, jti? })` | Signs an HS256 JWT (default expiry `24h`) |
| `verifyHs256(token, key)` | Verifies with `algorithms: ["HS256"]` (prevents algorithm substitution); throws on invalid or expired tokens |

## Shared vs per-app

| Shared (this core) | Per-app (each app's `auth.ts`) |
| --- | --- |
| `hashPassword`, `verifyPassword` (bcrypt cost 12) | payload shape (`role`, `activeOrgId`, …) |
| `encodeJwtSecret` (+ optional min length) | jti policy (none / randomBytes / randomUUID) |
| `signHs256`, `verifyHs256` (algorithm pinned) | null-vs-throw verify contract |
| | cookie name, sameSite, session storage model |

## Vendoring

Apps that build in their own Docker context (`COPY . .` from the app
directory) cannot import a package that lives outside that directory. For
those apps, copy `auth-core.ts` into the app (for example
`src/lib/auth-core.ts`) and mark the copy as generated. Keep this file as the
single source of truth: fix it here, re-copy it, then typecheck each app. A
security fix (algorithm pin, bcrypt cost) then reaches every app in one step.

Peer dependencies: `jose`, `bcryptjs`.

## Build and test

```bash
npm install      # also builds dist/auth-core.js + .d.ts (the package entry point)
npm run typecheck
npm test         # builds, then runs test/*.test.mjs with node --test
```

The tests cover the bcrypt cost and verification, the secret-length check,
sign/verify with claims and `jti`, rejection of another key, an expired token
and a tampered payload, and that an HS512 token is refused (algorithm pin).
