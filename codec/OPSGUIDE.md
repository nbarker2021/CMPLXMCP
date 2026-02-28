# OPSGUIDE

## Runtime
- Scope path: `CMPLX-DevLab/mcp_os/codec`
- Use local entrypoints and scripts in this directory before cross-scope tooling.

## Validation
- Run relevant tests/lints for changed files.
- For Python changes, run `python -m py_compile` on modified modules.
- Confirm docs and operational commands remain accurate.

## Incident Handling
- Capture failing command, error output, and impacted path.
- Roll forward with a minimal fix and re-run validations.
