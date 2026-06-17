#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the large-source-doc benchmark tier.

Each memory doc is a large (several KB) technical markdown document with one
critical answer buried mid-document, demonstrating that:

  - Full-replay tokens ≈ corpus_size (grows with doc count × doc size)
  - Artesian tokens ≈ top_k × chunk_size  (~bounded, ~1 200 tokens)

The corpus and benchmark are deterministic (fixed templates, no randomness).
Run this script before `cargo run -p artesian-bench -- --seed-corpus benchmarks/large-source-corpus`.
"""
from __future__ import annotations

from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]

# fmt: off
# Each entry: (slug, topic_label, answer_value, question, tag_csv)
TOPICS = [
    (
        "email-delivery-config",
        "email delivery retry interval",
        "45 seconds",
        "What is the SMTP retry interval for failed email delivery attempts?",
        "email,delivery,smtp,retry",
    ),
    (
        "cdn-origin-pull",
        "CDN origin pull timeout",
        "8 seconds",
        "What is the CDN origin pull timeout?",
        "cdn,infrastructure,timeout,performance",
    ),
    (
        "job-queue-concurrency",
        "maximum concurrent workers per job queue",
        "12 workers",
        "What is the maximum number of concurrent workers allowed per job queue?",
        "job-queue,workers,concurrency,background",
    ),
    (
        "db-connection-pool-large",
        "maximum database connection pool size for write replicas",
        "40 connections",
        "What is the maximum database connection pool size for write replicas?",
        "database,connection-pool,postgresql,writes",
    ),
    (
        "mobile-push-ttl",
        "mobile push notification TTL",
        "3 days",
        "What is the TTL for mobile push notifications?",
        "mobile,push,ttl,notifications",
    ),
    (
        "api-key-rotation",
        "mandatory API key rotation period",
        "90 days",
        "How often must API keys be rotated?",
        "api-keys,security,rotation,compliance",
    ),
]
# fmt: on

_PREAMBLE = """
## Background

The platform processes millions of events daily across multiple regions. Early in the
platform's history (2019–2021), configuration values were embedded directly in application
code, leading to inconsistent behaviour across services and environments. This document
captures the result of the 2022 standardisation effort, which established canonical values
enforced through the central configuration service.

Configuration drift incidents in Q3 2022 prompted the creation of this specification.
Engineers are expected to consult this document before changing any related settings.
Deviations require a formal change request reviewed by the platform architecture board.

Load testing performed in Q1 2023 using a traffic model derived from the 95th-percentile
production workload established the baseline values documented here. Tests were repeated
quarterly to verify that the values remain appropriate as the platform scales. The process
for quarterly review is documented in `docs/process/config-review.md`.

## Architecture Overview

The system is composed of three primary layers: the edge network, the application tier,
and the persistence tier. Each layer has its own resource limits and timeout budgets.
The values in this document apply to the application tier unless otherwise noted.

Cross-cutting concerns such as rate limiting, circuit breaking, and observability are
handled by the platform middleware rather than individual services. This ensures that
any change to the middleware stack is evaluated once, rather than requiring all consuming
services to update independently.

Service mesh configuration (Envoy) propagates updated values within 60 seconds of a
change to the central configuration service. Services that cannot tolerate a 60-second
propagation window must implement their own local override mechanism, documented in
`docs/platform/local-overrides.md`.

## Context and Motivation
"""

_CORE = """

## Core Configuration Decision

After evaluating performance benchmarks, load test results, and vendor recommendations,
the platform architecture board ratified the following canonical value on 2023-05-15:

**The {topic} is set to {answer}.**

This value balances resource utilisation against latency requirements for the
99th-percentile workload. It was validated through a 72-hour canary deployment in the
EU-West region before being rolled out globally. Changing this value requires a
performance impact assessment and sign-off from the platform lead.

The decision record for this value is filed under `decisions/2023-05-15-{slug}.md`.
Platform engineers are expected to link future change requests to this decision record
when proposing modifications.

## Implementation Details
"""

_IMPL = """
The configuration is applied via the central config service (`config.example.internal`).
Services that consume this value must use the SDK client rather than hardcoding the value
in their configuration files. The SDK client supports automatic refresh with a local cache
TTL of 60 seconds to minimise latency at read time.

Service teams are responsible for handling configuration refresh gracefully — no restart
should be required when the canonical value changes within the documented range.
Out-of-range changes trigger a controlled rolling restart via the deployment pipeline.

Integration tests in the `platform-tests` repository verify that every consuming service
reads the value from the config service rather than from a local override. These tests run
as part of the release gate and block deployment if any service is found to use a stale
value. The gating logic is implemented in `platform-tests/gate/config_gate_test.go`.

Observability for this value is provided by a dedicated Grafana panel on the
`Platform Configuration` dashboard. Alerts fire when the observed value deviates from
the documented canonical value by more than 20% for a sustained period of 5 minutes.

## Operational Notes

The SRE team reviews this value quarterly as part of the capacity planning cycle.
The review considers: (1) p99 latency trends, (2) error budget consumption, (3)
infrastructure cost, and (4) feedback from service teams. If the review concludes that
a change is warranted, the change is proposed to the architecture board with supporting
data from the review.

Historical canonical values:
- 2022-01: Initial standardisation (provisional, not published)
- 2022-07: First published value (pre-production validation only)
- 2023-05: Current value ratified after full-scale load testing
- Next scheduled review: 2024-Q1

## Cross-References

Related decisions and runbooks:
- `memory/slo-policy.md` — SLO commitments for this subsystem
- `memory/oncall-policy.md` — on-call rotation and escalation
- `docs/runbooks/{slug}.md` — step-by-step incident response procedures
- `docs/process/config-review.md` — quarterly review process
- `decisions/2023-05-15-{slug}.md` — original decision record
"""

_DISTRACTOR_BODY = """
This reference document covers general platform guidance related to {label}.
It is not an authoritative decision document; consult the specific decision files for
canonical values.

## Overview

The platform maintains a variety of configurable parameters across its subsystems.
This reference page provides context for engineers unfamiliar with the platform's
configuration model. Canonical values for all production parameters are stored in the
central configuration service and enforced via the SDK client.

## Configuration Model

All canonical values are stored in the central configuration service. Services must
use the SDK client to read values at startup and must implement graceful refresh.
Local overrides are permitted only in development environments; production deployments
enforce values from the central service without exception.

The central config service provides strong consistency guarantees for reads within
a single region. Cross-region propagation has a maximum latency of 30 seconds under
normal conditions and up to 5 minutes during a regional failover event.

## General Guidance

When evaluating whether a configuration value needs to change, engineers should gather
at least two weeks of performance data under realistic load conditions. Ad-hoc changes
based on single-incident evidence are discouraged; they tend to introduce instability.

Proposed changes to any configuration value documented in a `memory/` file must go
through the standard RFC process. The RFC template is in `docs/rfc-template.md`.
Engineers should expect a minimum review cycle of 5 business days for non-emergency
changes. Emergency changes may use the fast-track process described in
`docs/process/emergency-config-change.md`.

## Monitoring

Standard dashboards for platform configuration are available on the internal
observability platform. Alert thresholds are set to fire when observed behaviour
deviates from documented baselines. False positives should be reported to the platform
team via the `#platform-alerts` Slack channel.
"""


def _make_motivation(slug: str, topic: str) -> str:
    return f"""
The motivation for standardising the {topic} arose from three distinct incident types
observed between Q4 2021 and Q2 2022:

1. **Resource exhaustion**: services with no explicit bound consumed unbounded resources
   under sustained load, causing cascading failures across unrelated subsystems.
2. **Inconsistent behaviour**: different teams chose different values independently,
   making cross-service debugging and capacity planning extremely difficult.
3. **Undocumented changes**: ad-hoc adjustments made in production were not reflected in
   documentation, leading to drift between stated and actual behaviour.

The standardisation effort resolved all three issues by: (a) establishing a single
canonical value, (b) enforcing it via the SDK client, and (c) requiring all changes to
go through the formal RFC process documented in `docs/process/config-review.md`.

The initial value was selected by the platform architecture board after reviewing data
from three comparable platforms, two academic studies on workload characteristics, and
internal load test results from 2022-Q4. The selection criteria are documented in the
decision record filed under `decisions/2023-05-15-{slug}.md`.
"""


def make_large_doc(slug: str, topic: str, answer: str, tags_str: str) -> str:
    tags = [t.strip() for t in tags_str.split(",")]
    header = (
        f"---\n"
        f"type: decision\n"
        f"tags: [{', '.join(tags)}]\n"
        f"license: Apache-2.0\n"
        f"---\n\n"
        f"# {topic.title()} Configuration\n"
    )
    motivation = _make_motivation(slug, topic)
    core = _CORE.format(topic=topic, answer=answer, slug=slug)
    impl = _IMPL.format(slug=slug)
    return header + _PREAMBLE + motivation + core + impl


def make_distractor(slug: str) -> str:
    label = slug.replace("-", " ")
    return (
        f"---\n"
        f"type: reference\n"
        f"tags: [infrastructure, general]\n"
        f"license: Apache-2.0\n"
        f"---\n\n"
        f"# {label.title()} Reference\n"
        + _DISTRACTOR_BODY.format(label=label)
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the large-source-doc benchmark corpus"
    )
    parser.add_argument(
        "--out",
        default="large-source-corpus",
        help="output directory name under benchmarks/ (default: large-source-corpus)",
    )
    args = parser.parse_args()

    out = BENCH / args.out
    mem = out / "memory"
    dist = out / "distractors"
    mem.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)

    tasks = []
    for slug, topic, answer, question, tags_str in TOPICS:
        doc = make_large_doc(slug, topic, answer, tags_str)
        (mem / f"{slug}.md").write_text(doc, encoding="utf-8")
        # coherence_needle: a verbatim substring of the buried answer sentence in _CORE.
        # When small-to-big returns the full parent section, this needle is present in the
        # retrieved context. If only a 500-char chunk fragment is returned, it may be absent.
        tasks.append(
            {
                "id": slug,
                "difficulty": "hard",
                "question": question,
                "relevant_docs": [f"memory/{slug}.md"],
                "distractor_docs": [
                    "distractors/config-reference.md",
                    "distractors/platform-overview.md",
                ],
                "coherence_needle": f"The {topic} is set to {answer}.",
            }
        )
        print(f"  wrote memory/{slug}.md ({len(doc)} chars)")

    for slug in ["config-reference", "platform-overview", "sre-handbook", "arch-notes"]:
        doc = make_distractor(slug)
        (dist / f"{slug}.md").write_text(doc, encoding="utf-8")
        print(f"  wrote distractors/{slug}.md ({len(doc)} chars)")

    import json

    suite = {
        "suite": "seed-honest-v1",
        "notes": (
            "Large-source-doc tier: each memory doc is several KB with the critical "
            "answer buried mid-document. Demonstrates that Artesian retrieves the relevant "
            "chunk (~top_k × chunk_size ≈ 1 200 tokens) while full-replay grows linearly "
            "with corpus size."
        ),
        "tasks": tasks,
    }
    (out / "tasks.json").write_text(
        json.dumps(suite, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nGenerated {len(tasks)} tasks in {out}/")


if __name__ == "__main__":
    main()
