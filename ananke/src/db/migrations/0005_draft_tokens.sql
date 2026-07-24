-- 0005_draft_tokens: engine-reported speculative-decoding draft counts.
--
-- llama.cpp's `timings` object carries `draft_n` (tokens proposed by the
-- speculative draft during the request) and `draft_n_accepted` (how many of
-- those the target model accepted) whenever speculative decoding is active.
-- A healthy target/draft pairing accepts a substantial fraction; a sustained
-- acceptance of exactly zero across every drafting request is the signature
-- of a poisoned inference state (e.g. the 2026-07-24 all-NaN-logits wedge,
-- where the process kept serving HTTP 200s of garbage tokens). The
-- spec_collapse auto-restart watchdog queries these columns.
--
-- Both are nullable — engines without speculative decoding (or without a
-- `timings` object at all) leave them null, and such rows never count
-- toward the watchdog.

ALTER TABLE request_metrics ADD COLUMN draft_tokens INTEGER;
ALTER TABLE request_metrics ADD COLUMN draft_tokens_accepted INTEGER;
