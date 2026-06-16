---
type: decision
tags: [deployment, platform, blue-green]
license: Apache-2.0
timestamp: "2022-01-10T09:00:00Z"
---

# Deployment strategy — January 2022

**Deployment strategy** (superseded): The platform used **blue-green deployments**
with an idle standby environment. Traffic was cut over instantly; rollback required
swapping the load balancer target back within 5 minutes.

This approach was replaced in March 2024 by canary releases.
