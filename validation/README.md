# MCP OS Validation Framework

Comprehensive validation and testing framework for all MCP OS components.

## Structure

```
mcp_os/validation/
├── __init__.py                       # Package exports
├── README.md                         # This file
├── system_validator.py               # Core validation framework
├── universal_system_validator.py     # Universal System tests
├── agrm_mdhg_validator.py            # AGRM+MDHG integration tests
├── mcp_tools_validator.py            # MCP server tools tests
├── runner.py                         # CLI runner
├── diagnostics.py                    # System diagnostics
├── tests/                            # Additional test suites
├── diagnostics/                      # Diagnostic tools
└── benchmarks/                       # Performance benchmarks
```

## Usage

### Run All Validations

```bash
python -m mcp_os.validation.runner --all
```

### Run Specific Suites

```bash
# Universal System only
python -m mcp_os.validation.runner --universal

# AGRM+MDHG only
python -m mcp_os.validation.runner --agrm --mdhg

# Planet + Network
python -m mcp_os.validation.runner --planet --network

# MCP Server tools
python -m mcp_os.validation.runner --mcp
```

### Output Options

```bash
# Verbose output
python -m mcp_os.validation.runner --all --verbose

# JSON output
python -m mcp_os.validation.runner --all --json --output results.json

# Save to file
python -m mcp_os.validation.runner --all --output validation_report.json
```

## Test Coverage

### Universal System Tests (14 tests)

**Translator Tests:**

- `translator_text` - Text to geometric form

- `translator_code` - Code AST translation

- `translator_math` - Mathematical expressions

- `translator_data` - Data structures

**Crystal Tests:**

- `crystal_creation` - Create crystals from forms

- `crystal_resonance` - Resonance calculation

- `crystal_merge` - Crystal merging

**SNAP Tests:**

- `snap_transaction` - Transaction creation

- `snap_chain` - Chain operations

**Temporal Tests:**

- `temporal_coordinate` - Time-phase coordinates

- `hypothesis_generation` - Future-phase creation

- `memory_creation` - Past-phase memories

**Identity Tests:**

- `identity_registration` - Identity lifecycle

- `receipt_generation` - Speedlight receipts

- `provenance_audit` - Audit trails

### AGRM+MDHG Tests (21 tests)

**MDHG Cache Tests:**

- `mdhg_admission` - 24D vector admission

- `mdhg_quantization` - Quantization to 2D

- `mdhg_eviction` - Slot management

- `mdhg_multiscale` - Fast/med/slow layers

**CA Field Tests:**

- `ca_cell_creation` - 10-channel cells

- `ca_channel_updates` - Event-driven updates

- `ca_kernel_step` - Wolfram class dynamics

- `ca_multiscale` - Multi-scale fields

**AGRM Router Tests:**

- `agrm_node_creation` - Node management

- `agrm_sweep` - GR sweep ranking

- `agrm_zone_classification` - Shell assignments

- `agrm_path_building` - Path optimization

**Planet Tests:**

- `planet_creation` - Planet instantiation

- `planet_crystal_admission` - Crystal storage

- `planet_resonance_query` - Similarity search

- `planet_dynamics` - CA self-regulation

**Network Tests:**

- `network_creation` - Network setup

- `ribbon_creation` - Inter-planet ribbons

- `network_routing` - AGRM routing

## Validation Result Format

```python
{
    "name": "test_name",
    "component": "component_name",
    "status": "passed|failed|skipped",
    "message": "Human-readable description",
    "duration_ms": 42.5,
    "details": {...},      # Optional test-specific data
    "error": "..."         # Only for failed tests
}
```

## Exit Codes

- `0` - All tests passed

- `1` - One or more tests failed

## CI/CD Integration

```yaml
# .github/workflows/validation.yml
- name: Run MCP OS Validation
  run: |
    python -m mcp_os.validation.runner --all --json --output validation.json

- name: Check Results
  run: |
    if [ $(jq '.failed' validation.json) -gt 0 ]; then
      echo "Validation failed!"
      exit 1
    fi
```

## Adding New Tests

1. Create test method in appropriate validator class
2. Method must return `ValidationResult`
3. Add to `run_all_tests()` in same class
4. Test will be automatically picked up by runner

Example:

```python
async def _test_my_feature(self) -> ValidationResult:
    start = time.time()
    try:
        # Your test code
        assert something == expected

        return ValidationResult(
            name="my_feature",
            component="my_component",
            status="passed",
            message="Works correctly",
            duration_ms=(time.time() - start) * 1000
        )
    except Exception as e:
        return ValidationResult(
            name="my_feature",
            component="my_component",
            status="failed",
            message=str(e),
            duration_ms=(time.time() - start) * 1000,
            error=str(e)
        )
```

## Performance Benchmarks

The framework also includes performance benchmarks that measure:

- **Throughput**: Crystals/second admitted

- **Latency**: Query response times

- **Memory**: Per-planet memory footprint

- **Scalability**: Multi-planet performance

Run benchmarks:

```bash
python -m mcp_os.validation.benchmarks
```

## System Diagnostics

Real-time system health monitoring:

```bash
python -m mcp_os.validation.diagnostics
```

Checks:

- Python version compatibility

- Required packages installed

- Import paths correct

- Memory availability

- Disk space
