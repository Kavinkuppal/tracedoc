# tracedoc-server

Receives opt-in anonymized statistics from tracedoc. **Not deployed anywhere.** Data sharing is
behind a feature flag that stays off until the data terms are approved.

- `POST /v1/contributions`: validates the body against `tracedoc.share.payload.Contribution`
  (the same strict allow-list the client builds) and stores it.
- `DELETE /v1/installs/{install_id}`: removes every contribution from that install.

Privacy: no access log, no proxy-header parsing (`X-Forwarded-For` is ignored), no IP column,
and validation errors never echo submitted values. Rate limits are keyed by install ID.
HTTPS only: non-HTTPS requests get 403 (trusting only `X-Forwarded-Proto` from a TLS proxy).

Local testing:

```bash
pip install -e .. -e ".[dev]"
TRACEDOC_DB=postgresql:///tracedoc python -m tracedoc_server --allow-http
pytest            # starts a throwaway Postgres cluster (needs initdb/pg_ctl on PATH or PG_BIN)
```
