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

All SCoT artifacts in this stage must be **unit-level SCoT**. This means each artifact must decompose reasoning to the level of concrete implementation units (for example: functions, methods, handlers, validators, mappers, query builders, and test cases), not only file-level summaries.

All unit-level SCoT artifacts must also use **ReAct-style reasoning** (Reason + Act + Observe) to make decision-making explicit and verifiable.

Each SCoT artifact must be written using program-structure concepts from Structured Chain-of-Thought prompting:

1. **Input / Output Structure**
2. **Sequence Structure**
3. **Branch Structure**
4. **Loop Structure**

The purpose of these artifacts is to make the later implementation phase precise, ordered, and testable.

For each implementation unit, represent reasoning as a ReAct loop:

```text
Thought:
  - What is the next sub-problem to solve and why.

Action:
  - Concrete implementation action to perform.

Observation:
  - Expected result, validation signal, or feedback from the action.
```

Use as many ReAct iterations as needed per unit until the unit reaches a clear completion condition.

Use the following unit-level SCoT example as a reference for granularity and structure:

```text
Example:
An SCoT for the following function declaration:

def first_Repeated_Char(str):
"""
Write a python function to find the first repeated
character in a given string.
"""
Pass

Input: str: a string
Output: ch: a repeated character in str
1: for each character ch in str:
2: if ch appears more than once in str:
3: return ch
4: return None
```

## Core File-Writing Requirement

You must write your SCoT artifacts into `{{ output_path }}`.

The SCoT files must be located in the same directory structure where the eventual implementation files will be created.

For every planned implementation file, create a corresponding Markdown SCoT file.

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
- If the plan is ambiguous, mark the ambiguity explicitly in the relevant SCoT file.
- If the plan appears technically flawed, identify the issue and propose a pragmatic adjustment in the relevant SCoT file.
- Keep every SCoT artifact implementation-oriented, not merely a high-level summary.


## Task

Carefully analyze the PRD and implementation plan, derive the eventual implementation file structure, and then create one SCoT Markdown file for each eventual implementation artifact.

Each SCoT file must act as the blueprint for creating its corresponding source, test, or documentation file in a later step.

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
  README.md

SCoT files to create now:
  {{ output_path }}/src/config.ts.scot.md
  {{ output_path }}/src/services/user-service.ts.scot.md
  {{ output_path }}/src/routes/users.ts.scot.md
  {{ output_path }}/tests/user-service.test.ts.scot.md
  {{ output_path }}/README.md.scot.md
```

## 2. Write a Root Index File

Create this file:

```text
{{ output_path }}/SCOT_INDEX.scot.md
```

This file must summarize:

- The eventual implementation file structure
- The generated SCoT file structure
- The recommended implementation order
- Cross-file dependencies
- Global assumptions, risks, and unresolved clarifications

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

## 3. File-Level Implementation Sequence Structure

Break the eventual implementation of this file into an ordered sequence of concrete steps.

Use numbered steps.

Each step must include:

```text
Step <n>: <implementation action>
Purpose:
  - ...

Inputs:
  - ...

Outputs:
  - ...

Validation:
  - ...
```

The sequence must reflect dependency order within the file.

## 4. File-Level Branch Structure

Identify decision points, conditional behavior, error cases, and edge cases relevant to this file.

Use this format:

```text
Branch: <condition or decision point>
If:
  - <condition>
Then:
  - <expected behavior>
Else:
  - <fallback or alternative behavior>
Validation:
  - <how this branch should be tested>
```

Include branches for:

- Input validation
- Missing or invalid configuration
- Dependency failures
- Empty or malformed data
- Permission or environment issues, if applicable
- Feature-specific conditional behavior described in the PRD or plan

## 5. File-Level Loop Structure

Identify repeated processes relevant to this file.

Use this format:

```text
Loop: <repeated process>
For each:
  - <item being iterated>
Do:
  - <operation>
Stop condition:
  - ...
Validation:
  - ...
```

Include loops for:

- Iterating over entities, records, files, routes, jobs, or tasks
- Repeated validation
- Batch processing
- Retrying, polling, pagination, or scheduled workflows, if applicable
- Test-case generation across multiple components

## 6. File-Level Dependency Structure

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

## 7. Testing Notes for This File

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

## 8. Documentation Notes for This File

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

## 9. Risks, Assumptions, and Clarifications

List unresolved issues that may affect implementation of this file.

Use this format:

```text
Risk / Assumption: <name>
Description:
  - ...

Impact:
  - ...

Recommended handling:
  - ...
```

## 10. Final File-Level SCoT Blueprint

Conclude with a compact structured blueprint that a later code-generation step can follow.

Use this format:

```text
Input:
  - ...

Output:
  - ...

Sequence:
  1. ...
  2. ...
  3. ...

Branches:
  - If ... then ... else ...
  - If ... then ... else ...

Loops:
  - For each ... do ...

Validation:
  - ...

ReAct summary:
  - Thought: ...
  - Action: ...
  - Observation: ...

Ready for code generation:
  - Yes / No

If not ready:
  - Blocking clarification:
      - ...
```

---

# Final Output Rules

When you run this prompt:

1. Write the SCoT files to `{{ output_path }}`.
2. Create `{{ output_path }}/SCOT_INDEX.scot.md`.
3. Create one `*.scot.md` file for every eventual implementation file.
4. Preserve the final implementation directory structure.
5. Do not write source code, test code, or final documentation yet.
6. Do not output the SCoT only in chat.
7. Do not write outside `{{ output_path }}`.
8. The final chat response, if any, should only summarize the files written.

The generated output must be a file-based SCoT artifact set, not a single monolithic response.