"""FastAPI app. Validates contributions against tracedoc's own allow-list schema.

Privacy: no handler reads the client address, nothing logs it, and the store has no column
for it. Run with `python -m tracedoc_server`, which turns off uvicorn's access log and proxy
header parsing (see __main__.py).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from tracedoc.share.payload import Contribution

from .db import Store
from .ratelimit import RateLimiter

MAX_BODY_BYTES = 5_000_000
log = logging.getLogger("tracedoc_server")


def create_app(
    store: Store,
    *,
    allow_http: bool = False,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """`allow_http` is for local testing only; production must sit behind TLS."""
    app = FastAPI(title="tracedoc contributions", docs_url=None, redoc_url=None, openapi_url=None)
    limiter = limiter or RateLimiter()

    @app.middleware("http")
    async def https_only(request: Request, call_next):  # noqa: ANN001, ANN202
        # Behind a TLS-terminating proxy, trust only X-Forwarded-Proto, never X-Forwarded-For.
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        if scheme != "https" and not allow_http:
            return JSONResponse({"error": "https required"}, status_code=403)
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/v1/contributions", status_code=201)
    async def contribute(request: Request) -> Response:
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse({"error": "payload too large"}, status_code=413)
        try:
            contribution = Contribution.model_validate_json(body)
        except ValidationError as e:
            # Report field locations only; never echo submitted values back or into logs.
            fields = sorted({".".join(str(p) for p in err["loc"]) for err in e.errors()})
            return JSONResponse(
                {"error": "invalid payload", "fields": fields[:20]}, status_code=422
            )
        if not limiter.allow(str(contribution.install_id)):
            return JSONResponse({"error": "rate limited"}, status_code=429)
        stored = store.add(
            contribution.install_id,
            contribution.deletion_key_hash,
            contribution.schema_version,
            # The hash lives once, in the installs table; not repeated in every row.
            contribution.model_dump(mode="json", exclude={"deletion_key_hash"}),
        )
        if not stored:
            return JSONResponse({"error": "install key mismatch"}, status_code=403)
        log.info(
            "stored contribution schema=%s runs=%d",
            contribution.schema_version,
            len(contribution.runs),
        )
        return JSONResponse({"status": "stored"}, status_code=201)

    @app.delete("/v1/installs/{install_id}")
    async def delete_install(install_id: UUID, request: Request) -> Response:
        """Requires `Authorization: Bearer <deletion token>`; the install ID alone is not enough."""
        if not limiter.allow(f"delete:{install_id}"):
            return JSONResponse({"error": "rate limited"}, status_code=429)
        auth = request.headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return JSONResponse(
                {"error": "deletion token required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        stored_hash = store.deletion_key_hash(install_id)
        if stored_hash is None:
            return JSONResponse({"error": "nothing stored for this install"}, status_code=404)
        offered = hashlib.sha256(token.strip().encode("utf-8")).hexdigest()
        if not hmac.compare_digest(offered, stored_hash):
            return JSONResponse({"error": "invalid deletion token"}, status_code=403)
        removed = store.delete_install(install_id)
        log.info("deleted install data rows=%d", removed)
        return JSONResponse({"status": "deleted", "rows": removed})

    return app
