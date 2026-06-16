---
type: decision
tags: [payments, idempotency, stripe, multi-session-synthesis]
license: Apache-2.0
---

# Payment idempotency key retention

All payment requests must include a Stripe idempotency key. The platform stores
idempotency keys for **24 hours** after the initial request. Duplicate requests
within this window are deduplicated automatically; requests after 24 hours are
treated as new charges.
