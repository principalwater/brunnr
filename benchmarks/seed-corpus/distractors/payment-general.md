---
type: reference
tags: [payments, overview]
license: Apache-2.0
---

# Payment system overview

The payment system processes customer transactions via Stripe. All charges are
PCI-compliant; card data never touches our servers directly. Refunds are
processed asynchronously and typically settle within 5-10 business days.

See specific decision records for timeout and idempotency configuration.
