# AGENTS.md

## 1. Purpose

This repository is built for reproducible, modular, and scalable experimentation.

Agents modifying this repository must preserve the following properties:

- Clear separation of responsibilities.
- Reproducible experiments.
- Explicit and stable data contracts.
- Resumable long-running workflows.
- Cross-platform execution.
- Testability.
- Extensibility without unnecessary rewrites.
- Clear separation between core logic and interfaces such as CLI or API.

Treat this file as an architectural contract, not as optional guidance.

---

## 2. General Rules

### MUST

- Read this file before modifying the repository.
- Inspect the existing architecture and conventions before introducing new code.
- Keep changes scoped to the requested task.
- Preserve existing behavior unless a behavior change is explicitly intended.
- Prefer simple, explicit designs over unnecessary abstraction.
- Use existing shared utilities and contracts when they already satisfy the requirement.
- Add tests for new or changed behavior.
- Keep configuration, implementation, and experiment results reproducible.
- Make important assumptions explicit in code, configuration, or documentation.

### MUST NOT

- Introduce unrelated refactors while implementing a feature.
- Duplicate existing shared logic without a clear reason.
- Silently change an established data contract.
- Hardcode environment-specific paths, devices, or experiment parameters in reusable logic.
- Hide errors with broad exception handling or silent fallbacks.
- Put substantial reusable logic inside CLI entrypoints, notebooks, or ad-hoc scripts.

---

## 3. Architecture and Separation of Responsibilities

Each module should have a clear primary responsibility.

A stage that performs a distinct part of a workflow should remain independently callable, testable, and replaceable.

Do not create large files that combine multiple independent stages or responsibilities.

Prefer this conceptual separation:

```text
input / preparation
        ↓
processing stage
        ↓
artifact
        ↓
next stage
        ↓
retrieval / inference
        ↓
evaluation
```

The exact number and names of modules may evolve with the project.

### Dependency direction

Prefer dependencies toward lower-level, reusable abstractions.

Higher-level workflows may depend on shared utilities and contracts, but shared utilities should not depend on higher-level pipeline stages.

Avoid circular dependencies between stages.

### Core logic vs interfaces

Core functionality must not depend on a specific interface.

CLI commands, APIs, notebooks, and other interfaces should call reusable core functions rather than contain the implementation themselves.

This allows the same functionality to be used from:

- CLI commands
- tests
- notebooks
- future APIs
- other internal workflows

---

## 4. Functions and Code Quality

Reusable functions must have a single clear responsibility.

Functions should generally:

- have explicit type annotations;
- have meaningful names;
- have predictable inputs and outputs;
- avoid hidden global state;
- avoid side effects unless those side effects are part of the function's responsibility.

### Function documentation

Reusable functions should document:

1. One-line summary.
2. Expected input.
3. Returns.
4. Raises.

Use this structure:

```python
def example_function(...) -> ReturnType:
    """
    One-line summary.

    Expected input:
        - ...

    Returns:
        - ...

    Raises:
        - ValueError: ...
    """
```

Do not add documentation that merely restates the function name. Document assumptions, contracts, and failure conditions that matter to callers.

---

## 5. Shared Helpers and Utilities

Reusable helper logic belongs in shared modules rather than in individual pipeline entrypoints.

Examples include:

- configuration loading;
- path handling;
- device resolution;
- logging;
- validation;
- serialization;
- hashing;
- checkpoint handling;
- common retrieval operations;
- common ranking operations.

Before creating a helper, check whether an equivalent utility already exists.

Avoid creating multiple helpers that solve the same problem with slightly different behavior.

Shared helpers should remain domain-appropriate. Do not create a generic abstraction merely to reduce the number of lines in one file.

---

## 6. Data Contracts

Stages must communicate through explicit and stable data contracts.

A data contract defines:

- required fields;
- field meanings;
- types;
- valid ranges or constraints where applicable;
- ordering guarantees where applicable;
- relationships between fields;
- schema version.

Do not rely on undocumented dictionary keys, positional assumptions, or filename conventions as the only contract between stages.

Prefer explicit structures such as:

- dataclasses;
- typed models;
- typed dictionaries;
- validated schemas;
- clearly defined serialized formats.

### Contract changes

Changing a shared contract is an architectural change.

When changing one:

- identify affected producers and consumers;
- update validation and tests;
- update documentation;
- preserve backward compatibility when practical, or explicitly version the schema.

---

## 7. Artifact Design

Outputs should be self-describing enough to remain useful outside the process that produced them.

An artifact should carry, directly or through associated metadata, the information needed to understand:

- what it contains;
- which schema it follows;
- which stage produced it;
- which input it came from;
- relevant model or processing configuration;
- relevant version information;
- creation or run identity.

Do not assume that the filename alone is sufficient metadata.

Prefer immutable, run-specific outputs over silently overwriting previous experiment results.

Overwriting existing artifacts must require explicit intent when there is a risk of data loss.

---

## 8. Configuration and CLI

Configuration should be driven by files rather than hardcoded constants.

YAML is the preferred human-editable configuration format unless an existing project convention requires otherwise.

Use a clear precedence model:

```text
base/default configuration
        ↓
environment or experiment configuration
        ↓
explicit CLI overrides
```

CLI arguments and flags should make experiment-sensitive parameters easy to override without editing source code.

Examples of parameters that should generally be configurable include:

- input/output locations;
- model selection;
- device;
- batch size;
- limits or subsets;
- ranking parameters;
- experiment options.

Do not expose every internal implementation detail as a CLI flag. Prefer meaningful, experiment-relevant controls.

The effective configuration should be inspectable before or during a run.

Support a dry-run or equivalent validation mode for workflows where doing so is useful.

---

## 9. Reproducibility

Every meaningful experiment should be reproducible from:

- source code version;
- configuration;
- relevant input/data version;
- execution environment;
- command or invocation parameters.

Record sufficient metadata to identify a run and reproduce its conditions.

Avoid hidden configuration from:

- undeclared globals;
- machine-specific paths;
- undocumented environment assumptions;
- manually edited source files;
- implicit notebook state.

Randomness must be controlled when reproducibility requires it.

---

## 10. Resumability and Checkpointing

Long-running or expensive stages should support checkpointing and resumption.

A resumable stage should distinguish between:

- completed work;
- incomplete work;
- failed work;
- invalid or corrupted artifacts.

Do not treat file existence alone as proof that a unit of work completed successfully.

Checkpoint state should be explicit and recoverable.

Whenever practical:

- process work in independently recoverable units;
- persist progress incrementally;
- make interrupted runs restartable;
- avoid recomputing completed work unnecessarily.

---

## 11. Safe Writes and Data Integrity

Artifacts should not be exposed as valid final outputs until they have been successfully written.

Prefer atomic or otherwise integrity-safe write patterns for important artifacts.

A failed or interrupted process must not silently leave a partially written artifact that looks valid.

Validate important outputs before marking work complete.

---

## 12. Cross-Platform Compatibility

Code should work across supported operating systems unless a component explicitly requires platform-specific behavior.

### File paths

Use platform-independent path APIs such as `pathlib`.

Do not concatenate filesystem paths manually.

Do not hardcode:

- `/`
- `\`
- drive letters
- user-specific home directories

unless handled through an appropriate platform abstraction.

### Runtime assumptions

Do not assume:

- a specific shell;
- a specific GPU vendor;
- CUDA availability;
- a specific accelerator backend;
- Unix-only utilities;
- Unix process semantics.

Platform-specific behavior must be isolated and have an appropriate fallback or explicit requirement.

Code involving multiprocessing or subprocesses must respect platform-specific process behavior.

---

## 13. Logging and Error Handling

Use structured, meaningful logging rather than scattered debug prints in reusable code.

Logs should make it possible to understand:

- what stage is running;
- which run is executing;
- progress;
- important decisions;
- warnings;
- failures;
- output locations;
- performance where useful.

Errors must be visible.

Do not use:

```python
except Exception:
    pass
```

or equivalent silent failure patterns.

Catch exceptions only when there is a meaningful recovery or when adding useful context before re-raising.

Fail early when an invalid input or inconsistent state would make the produced result unreliable.

---

## 14. Validation

Validate inputs and configuration before expensive processing starts whenever practical.

Examples:

- required paths exist;
- required files are readable;
- schemas are compatible;
- parameters are valid;
- output locations are writable;
- requested resources are available.

Prefer clear validation errors over obscure failures deep inside a pipeline.

---

## 15. Testing

Testing is part of implementation, not a final optional step.

Use an appropriate combination of:

### Unit tests

Test small, deterministic components such as:

- parsing;
- validation;
- transformations;
- ranking;
- scoring;
- path handling;
- configuration behavior;
- serialization;
- checkpoint logic.

### Integration tests

Test interactions between stages using a small controlled dataset or fixture.

### Regression tests

Preserve known-good behavior and previously verified results where the behavior is intended to remain stable.

Tests should be fast enough to run frequently during development.

Expensive tests should be clearly separated from the default fast test suite.

---

## 16. Experiment Workflow

A new experiment must not destabilize a known-good implementation.

Use this cycle:

```text
change
  ↓
test
  ↓
evaluate
  ↓
record result
  ↓
commit
  ↓
next change
```

Do not stack several unrelated experimental changes before evaluating them.

Keep experiments isolated through configuration, run directories, branches, or another explicit mechanism.

Never modify a known-good baseline merely to test an experiment.

An experiment that produces a small improvement must be evaluated against the same evaluation procedure and baseline before being considered a verified improvement.

---

## 17. Small-Scale Validation Before Expensive Runs

Before launching a large or expensive job, provide a practical way to validate the workflow on a small subset.

Prefer explicit controls such as:

- selected input IDs;
- sample counts;
- limited batches;
- dry-run mode.

A small successful run should verify:

- input discovery;
- model loading;
- output format;
- schema compatibility;
- checkpoint behavior;
- downstream compatibility.

Only then should the full workload be launched.

---

## 18. Retrieval, Ranking, and Evaluation

Separate these concerns:

```text
candidate generation
        ↓
scoring / ranking
        ↓
post-processing
        ↓
output formatting
        ↓
evaluation
```

Shared retrieval primitives should be reusable across different query or task types where their semantics are compatible.

Do not duplicate the same search or ranking implementation for each task merely because the entrypoints differ.

Evaluation must be independent from retrieval implementation.

Evaluation code should consume predictions and ground truth through explicit contracts rather than inspecting retrieval internals.

Support task-specific scoring logic where required, while keeping common ranking/aggregation behavior shared.

Do not assume that a metric, score, or evaluation result represents the entire system unless the evaluation definition explicitly says so.

---

## 19. Extensibility

New features should be attachable without rewriting unrelated stages.

When adding functionality:

- identify the smallest appropriate extension point;
- reuse existing contracts;
- add a dedicated module when the functionality is independently meaningful;
- place reusable helpers in shared modules;
- avoid embedding new features into unrelated pipeline files.

Prefer composition over large conditional blocks such as:

```python
if mode == "...":
    ...
elif mode == "...":
    ...
elif mode == "...":
    ...
```

When a growing set of implementations shares a common contract, consider a clear interface or registry rather than continually expanding one monolithic function.

Do not introduce plugin systems or elaborate frameworks unless the project actually benefits from them.

---

## 20. API Readiness

Core outputs and functions should be designed so that a future API can call them without invoking command-line scripts as a subprocess.

Prefer structured return values that can be serialized cleanly.

Keep API-facing schemas separate from internal implementation details where necessary.

Do not make CLI stdout the only way to access a result.

Human-readable logs and machine-readable results should remain distinguishable.

---

## 21. Notebooks and Ad-hoc Scripts

Notebooks and temporary scripts are for exploration, visualization, and debugging.

Production behavior must live in importable source modules.

A notebook should not be the only place where an important pipeline step exists.

Useful experimental logic discovered in a notebook should be moved into tested source code before becoming part of the supported workflow.

---

## 22. Documentation

Document decisions that future contributors or agents would otherwise have to rediscover.

Useful documentation includes:

- architecture decisions;
- data contracts;
- configuration conventions;
- experiment methodology;
- important assumptions;
- expected inputs and outputs;
- reproducible commands.

Avoid documenting implementation details that are obvious from clear code.

Documentation should describe current behavior, not aspirational behavior.

---

## 23. Agent Change Discipline

When asked to modify the repository:

1. Read this file.
2. Inspect the relevant existing code and contracts.
3. Identify the smallest set of files that need to change.
4. Implement the change without unrelated refactoring.
5. Run relevant tests and validation.
6. Check formatting, types, and imports where applicable.
7. Verify that existing contracts remain intact.
8. Report what changed and what was validated.

If an architectural change is necessary, make it explicit rather than silently redesigning the repository.

Do not modify unrelated files merely to make the code look cleaner.

---

## 24. Definition of Done

A change is complete when applicable requirements have been satisfied:

- implementation is complete;
- interfaces and contracts are clear;
- configuration is supported where appropriate;
- relevant CLI overrides exist where useful;
- reusable functions have type annotations and documentation;
- tests cover new or changed behavior;
- logging and error handling are appropriate;
- outputs are valid and discoverable;
- expensive work is safely restartable when applicable;
- existing behavior has been checked for regression;
- the change can be reproduced from its documented invocation.

A change is not complete merely because it runs successfully once on one machine.
