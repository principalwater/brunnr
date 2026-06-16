---
type: decision
tags: [deployment, platform, canary, temporal-ordering]
license: Apache-2.0
timestamp: "2024-03-01T09:00:00Z"
---

# Deployment strategy — updated March 2024

**Current deployment strategy**: The platform uses **canary releases** with a 5% traffic
split to the new version for 30 minutes, followed by a full rollout if the error rate stays
below 0.1%.

This supersedes the blue-green approach used prior to 2024. The canary model was adopted after
blue-green proved costly to maintain in multi-region active-active setups.
