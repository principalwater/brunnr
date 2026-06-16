---
type: decision
tags: [payments, timeout, stripe, multi-session-synthesis]
license: Apache-2.0
---

# Payment service timeout

The payment service enforces a **30-second timeout** on all Stripe API calls.
Requests that exceed this timeout are marked as failed and queued for manual
review. The timeout applies to both charge creation and refund requests.
