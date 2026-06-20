# SPDX-License-Identifier: Apache-2.0
"""Artesian — memory control plane for agent loops (Python bindings).

The native core is implemented in Rust and exposed via ``artesian._artesian``. This package is the
thin, Pythonic surface over it. Import only what you need — the LEGO principle applies here too.

Status: scaffold. The native surface is minimal until the Rust core (transactional substrate + ACC)
stabilizes; see the project roadmap.
"""

from ._artesian import Memory, version

__all__ = ["Memory", "version"]
