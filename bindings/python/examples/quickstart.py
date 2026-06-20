# SPDX-License-Identifier: Apache-2.0
"""Quickstart sketch for the Artesian Python bindings (scaffold).

Run after building locally with ``maturin develop`` in bindings/python.
The recall/commit surfaces are placeholders today; see the roadmap.
"""

import artesian

print("artesian bindings version:", artesian.version())

mem = artesian.Memory()

# Recall a high-signal slice for the current step (placeholder until the core is wired).
context = mem.recall("what did we decide about the auth refactor?")
print("recalled:", context)

# Commit a durable learning through the qualify-gate (placeholder until the core is wired).
committed = mem.commit("Decision: auth tokens rotate every 24h; see ADR-014.")
print("committed:", committed)
