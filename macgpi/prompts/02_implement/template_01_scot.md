# Software Implementation Structured Chain-of-Thought Planning Prompt

You are an expert software architect preparing to implement a detailed software plan into code.

Your current task is **only the first stage**: construct Structured Chain-of-Thought artifacts, also called SCoT artifacts, that will later be used to guide implementation.

You **must write files**.

However, you must **not** write production source code yet. Instead, write SCoT planning artifacts to the exact places where the eventual implementation files will go, using an SCoT-specific Markdown filename for each planned implementation file.

## Input

- **PRD**: `{{ system_prd }}`

- **IMPLEMENTATION PLAN**: `{{ implementation_plan }}`

## Goal

Generate structured implementation-reasoning artifacts that translate the PRD and implementation plan into code-oriented solving processes.

The implementation plan contains a top-level `type` field:

- `modification`: an existing project must be changed to fix an issue or add an enhancement.
- `greenfield`: a new project must be built from the ground up.

You must adapt the SCoT artifact set to this type.

For `modification` plans:
- Treat the existing project under `{{ output_dir }}` as authoritative.
- Inspect the existing repository structure before deciding which SCoT files to create.
- Create SCoT artifacts only for files that must be created or modified to satisfy the PRD and implementation plan.
- For existing files, plan surgical edits only. Preserve unrelated behavior, public interfaces, formatting style, and tests.
- Do not create broad architecture, README, configuration, or documentation SCoT artifacts unless the plan explicitly requires those files.
- Include regression-test SCoT artifacts when a bug fix or behavioral change should be protected by tests.
- Every SCoT artifact for an existing file must include a **Preservation Boundary** section that states what must remain unchanged and where edits are allowed.
- If a planned change can be made by editing an existing file, prefer a SCoT artifact for that existing file over creating replacement files or parallel implementations.

For `greenfield` plans:
- Derive the full project structure required by the PRD and implementation plan.
- Include source files, tests, configuration files, scripts, and documentation files where needed for a runnable, maintainable project.
- Create SCoT artifacts for all eventual implementation artifacts required to build the project from the ground up.
- Whole-file creation is expected for new project files, but each SCoT artifact must still define concrete units, tests, and dependencies.

All SCoT artifacts in this stage must be **unit-level SCoT**. This means each artifact must decompose reasoning to the level of concrete implementation units (for example: functions, methods, handlers, validators, mappers, query builders, and test cases), not only file-level summaries.

All unit-level SCoT artifacts must also use **ReAct-style reasoning** (Reason + Act + Observe) to make decision-making explicit and verifiable.

Each SCoT artifact must be written using program-structure concepts from Structured Chain-of-Thought prompting:

1. **Input / Output Structure**
2. **Structure (Sequence structure, Branch structure, Loop structure)**

The purpose of these artifacts is to make the later implementation phase precise, ordered, and testable.

For each implementation unit, provide a concise implementation trace, representing a classic ReAct loop:

Reason:
  - The implementation choice to make and why, without private chain-of-thought.

Act:
  - Concrete implementation action to perform.

Observation:
  - Expected result, test signal, or feedback from the action.

Use as many ReAct iterations as needed per unit until the unit reaches a clear completion condition.

Use the following unit-level SCoT example as a reference for granularity and structure:

```text
Unit: first_Repeated_Char(str):
Description: A python function to find the first repeated character in a given string.

Input: str: a string
Output: ch: a repeated character in str
1: for each character ch in str:
2:   if ch appears more than once in str:
3:     return ch
4: return None
```

## Core File-Writing Requirement

You must write your SCoT artifacts into `{{ output_dir }}`.

The SCoT files must be located in the same directory structure where the eventual implementation files will be created.

For every planned implementation file, create a corresponding Markdown SCoT file. The meaning of "planned implementation file" depends on `implementation_plan.type`:

- For `modification`, this means only the existing files that must be modified and the new files that are directly required by the requested fix or enhancement.
- For `greenfield`, this means every file needed for the new project.

Use this naming convention:

```text
<eventual-file-name>.<eventual-extension>.scot.md
```

For example:

```text
Eventual implementation file:
  src/auth/session.ts

SCoT file to write now:
  src/auth/session.ts.scot.md
```

Another example:

```text
Eventual implementation file:
  tests/auth/session.test.ts

SCoT file to write now:
  tests/auth/session.test.ts.scot.md
```

If the implementation plan includes documentation files, create SCoT files for those documentation files too.

For example:

```text
Eventual documentation file:
  README.md

SCoT file to write now:
  README.md.scot.md
```

## Important Constraints

- You **must write files**.
- Write only SCoT Markdown files in this stage.
- Do **not** write final source code.
- Do **not** write the final implementation files yet.
- Do **not** write test code yet.
- Do **not** write final documentation yet.
- Do **not** skip functionality from the PRD or implementation plan.
- Do **not** invent requirements that are not supported by the PRD or plan.
- Respect `implementation_plan.type` when deciding the artifact scope.
- For `modification`, do **not** plan unrelated rewrites, broad generated scaffolding, or whole-file replacement when a localized edit is sufficient.
- For `greenfield`, do **not** under-specify project setup, tests, or documentation required for a complete runnable project.
- If the plan is ambiguous, mark the ambiguity explicitly in the relevant SCoT file.
- If the plan appears technically flawed, identify the issue and propose a pragmatic adjustment in the relevant SCoT file.
- Keep every SCoT artifact implementation-oriented, not merely a high-level summary.


## File Modification Types

For each created SCoT file:

### Greenfield
- Create complete SCoT file with all required content
- No existing code to preserve

### Modification 
- CRITICAL: Do NOT rewrite entire file
- Show context (5-10 lines before/after each change)
- Preserve all unrelated code
- Use diff format examples
- Include a Preservation Boundary section for every existing target file
- Include a Testing Notes section that explains the targeted regression or behavior check for the changed units


### Validation Checklist
- [ ] File size is reasonable (not truncated)
- [ ] All imports are present
- [ ] No repeated/corrupted blocks


## Task

Carefully analyze the PRD and implementation plan, derive the eventual implementation file structure, and then create one SCoT Markdown file for each eventual implementation artifact.

Each SCoT file must act as the blueprint for creating its corresponding source, test, or documentation file in a later step.

Before writing SCoT files, determine the plan type from the top-level `type` field in the implementation plan.

If the type is `modification`, first identify the existing files that are relevant to the requested issue or enhancement. The SCoT index must explicitly distinguish:

- Existing files to modify
- New files to add
- Tests to add or update
- Files intentionally left unchanged
- Private planning or generated artifacts that must not become final implementation output
- The minimal-change boundary for the patch

If the type is `greenfield`, derive the complete initial project structure. The SCoT index must explicitly distinguish:

- Source files
- Test files
- Configuration and metadata files
- Documentation files
- Scripts or entry points

---

# Required Output Behavior

## 1. Derive the Eventual Implementation File Structure

First, infer the file structure that the final implementation will require.

You must include source files, test files, configuration files, scripts, and documentation files where applicable.

Then, for each eventual file, write exactly one SCoT file in the corresponding path.

Example:

```text
Eventual implementation structure:
  src/config.ts
  src/services/user-service.ts
  src/routes/users.ts
  tests/user-service.test.ts

SCoT files to create now:
  {{ output_dir }}/src/config.ts.scot.md
  {{ output_dir }}/src/services/user-service.ts.scot.md
  {{ output_dir }}/src/routes/users.ts.scot.md
  {{ output_dir }}/tests/user-service.test.ts.scot.md
  {{ output_dir }}/README.md.scot.md
```

## 2. Write a Root Index File

Create this file:

```text
{{ output_dir }}/SCOT_INDEX.scot.md
```

This file must summarize:

- The eventual implementation file structure
- The generated SCoT file structure
- The recommended implementation order
- Cross-file dependencies
- Global assumptions, risks, and unresolved clarifications
- The implementation plan `type`
- For `modification`, the minimal-change boundary and the existing files that must be preserved
- For `greenfield`, the complete project scaffold required for a runnable first version
- For `modification`, any generated SCoT or planning artifacts that are implementation aids only and should not be treated as product files

The index file must not contain final source code.

## 3. Write Per-File SCoT Artifacts

Each per-file SCoT artifact must use the structure below.

Within each file artifact, include unit-level SCoT breakdowns for the concrete implementation units that will exist in that file.

Examples of units to cover when applicable:

- Source files: functions, classes, methods, handlers, service operations, utility helpers
- Test files: test groups, individual test cases, fixtures, setup/teardown helpers
- Config/script files: loaders, validators, command steps, environment resolution logic

For each unit, include at least one ReAct iteration and preferably multiple iterations when there are dependencies, branches, or non-trivial validation steps.

---

# Per-File Structured Chain-of-Thought Artifact Template

Each generated `*.scot.md` file must contain the following sections.

## 1. Target File

```text
Eventual file:
  <relative/path/to/final/file>

SCoT file:
  <relative/path/to/final/file>.<extension>.scot.md

Artifact type:
  <source | test | config | script | other>
```

## 2. Requirement Intake Structure

Describe the implementation problem for this specific file in terms of inputs, outputs, constraints, and success criteria.

Use this format:

```text
Input:
  - PRD requirements relevant to this file:
      - ...
  - Implementation plan requirements relevant to this file:
      - ...
  - Upstream dependencies:
      - ...

Output:
  - Responsibilities of the eventual file:
      - ...
  - Public interfaces, exports, routes, schemas, or commands:
      - ...
  - Downstream consumers:
      - ...

Constraints:
  - ...
```

## 3. Preservation Boundary

This section is required for every SCoT artifact when `implementation_plan.type` is `modification`.

For existing files, describe the exact boundary of the intended edit. For new files added by a modification plan, explain why the new file is necessary and what existing behavior it must not disturb.

Use this format:

```text
Plan type:
  modification

Target status:
  <existing file to modify | new file required by modification>

Must preserve:
  - Existing public interfaces:
      - ...
  - Existing behavior:
      - ...
  - Existing tests, compatibility expectations, or documented behavior:
      - ...
  - Existing style, framework conventions, and dependency boundaries:
      - ...

Allowed change area:
  - Functions, classes, sections, tests, or configuration entries:
      - ...

Do not change:
  - ...

Reason a new file is required, if applicable:
  - ...
```

For `greenfield` plans, write:

```text
Plan type:
  greenfield

Preservation boundary:
  Not applicable because this is a new project file.
```

## 4. Unit-Level Implementation Sequence Structure

A **unit** is a concrete implementation item that will exist inside this file: a function, method, class, handler, schema definition, validator, query builder, or test case. Do not treat the file as a whole as a single unit.

Break the eventual implementation of this file into a set of units, with each unit having their own SCoTs.

Use numbered steps.

Each unit SCoT must include:

```text
Unit: ...
Description: ...

Input: ...
Output: ...
1: ...
2: ...
...

Reasoning:
1: Reason: ...
1: Act: ...
1: Observe ...

2: Reason: ...
2: Act: ...
2. Observe: ...

...
```

## 5. File-Level Dependency Structure

Describe how this file depends on and supports other files.

Use this format:

```text
Component or file:
  <relative/path>

Requires:
  - ...

Provides:
  - ...

Implementation order:
  - Must be implemented before:
      - ...
  - Must be implemented after:
      - ...
```

## 6. Testing Notes for This File

Describe how this file should be tested later.

Use this format:

```text
Test Group: <name>
Purpose:
  - ...

Covers:
  - ...

Test cases:
  1. Given ...
    When ...
    Then ...

Required mocks or fixtures:
  - ...
```

For test files, describe the tests that the eventual test file will contain.

For source files, describe which test files should validate this source file.

For `modification` plans, testing notes must focus on targeted regression coverage and preservation of nearby existing behavior.

For `greenfield` plans, testing notes must cover the core workflow or component responsibility introduced by the file.

## 7. Documentation Notes for This File

Identify documentation that should exist for the eventual file.

Use this format:

```text
Documentation Item: <name>
Audience:
  - ...

Must explain:
  - ...

Examples required:
  - ...
```

---

# Final Output Rules

When you run this prompt:

1. Write the SCoT files to `{{ output_dir }}`.
2. Create `{{ output_dir }}/SCOT_INDEX.scot.md`.
3. Create one `*.scot.md` file for every eventual implementation file.
4. Preserve the final implementation directory structure.
5. Do not write source code, test code, or final documentation yet.
6. Do not output the SCoT only in chat.
7. Do not write outside `{{ output_dir }}`.
8. The final chat response, if any, should only summarize the files written.

The generated output must be a file-based SCoT artifact set, not a single monolithic response.
