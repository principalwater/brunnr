<!-- SPDX-License-Identifier: Apache-2.0 -->

# Artesian — Python bindings (scaffold)

Write once in Rust, run from Python. These bindings expose the same audited Artesian core to Python
via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs), so Python users run the native,
in-process core — no second implementation, no per-call serialization tax. See
[Why Rust](../../docs/why-rust.md) and [Composability](../../docs/composability.md).

> **Status: scaffold.** The native surface is intentionally minimal until the core (transactional
> substrate + ACC) stabilizes. `recall` / `commit` are placeholders today. The structure and the
> write-once / run-many path are in place; the surface grows as the core lands.

## Build locally

```bash
pip install maturin
cd bindings/python
maturin develop            # builds the native module into your current venv
python examples/quickstart.py
```

## Use (target shape)

```python
import artesian                  # import only what you need (LEGO)

mem = artesian.Memory()
context = mem.recall("what did we decide about X?")   # high-signal slice, small-to-big
mem.commit("Decision: ...")                           # qualify-gate + commit into the CCS
```

## Notes

- The native module is `artesian._artesian`; the Pythonic surface lives in `python/artesian`.
- The PyPI distribution name (`artesian`) is to be confirmed before first publish.
- Detached from the root cargo workspace on purpose (built and versioned independently with maturin).
