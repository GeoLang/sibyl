# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- 2026-08-25: **an agora feed token is refused as a session bearer**. `subject`
  already rejected `token_use` and `geolang_use`, and now rejects `agora_use`
  with any value. agora mints a long lived feed token signed with the same
  `PLATFORM_JWT_SECRET` and carrying no `role`, so before this it opened
  sessions owned by the feed's uuid.

- 2026-08-25: **jsonwebtoken 9 to 11 on the `aws_lc_rs` backend**, the same
  crypto crate reqwest's rustls already pulls in. Only HS256 `encode`,
  `decode`, `from_secret` and `Validation::default()` are used, and 11 keeps
  `validate_aud` on by default, so a token carrying `aud` is still refused.

### Added

- 2026-08-25: **sessions have an owner**. sibyl reads `PLATFORM_JWT_SECRET`, the
  same HS256 platform secret every other service validates with, and refuses to
  start without it unless `SIBYL_ALLOW_UNAUTHENTICATED` is set to `1`, `true`,
  `yes` or `on`, which logs one line saying sessions are unowned. The verified
  `sub` of the bearer owns every session that caller creates: it is stored in a
  new nullable `subject` column, added by an idempotent `ALTER TABLE` on open.
  `GET /sessions` lists only the caller's own, and the active session is per
  subject, so creating or activating one no longer switches the session someone
  else is reading. `activate`, `rename`, `delete` and `POST
  /sessions/{id}/messages` answer 404 for a session belonging to another
  subject, the same answer a missing id gets, so ids cannot be probed. `POST
  /runs` takes the subject from `user_token`: a run with no `thread_id` uses the
  caller's own active session, and a `thread_id` naming someone else's session
  fails as not found. Every `/sessions` route and `/runs` answer 401 with
  `{"error": ...}` when the token is missing or invalid. Sessions written before
  the column keep a NULL subject and are reachable only with the gate off.
  A token carrying `token_use` or `geolang_use` is refused: those are geolang's
  scoped tool credential and its `/mcp` token, and the executor that runs
  caller-written code holds the tool ones.
