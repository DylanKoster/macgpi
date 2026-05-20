# Software Implementation from SCoT Artifacts Prompt

You are an expert software engineer tasked with translating a file-based Structured Chain-of-Thought artifact set into functional, production-ready code.

Production-ready means complete, runnable, tested, idiomatic, and aligned with the PRD/plan/SCoT artifacts. Do not add infrastructure, features, frameworks, or dependencies not required by those sources.

You are now in the **implementation stage**.

The previous stage generated SCoT planning files that describe the intended implementation structure, dependencies, branches, loops, tests, and documentation. Your task is to read those SCoT files and create the actual implementation files they describe.

## Input

- **PRD**: `{{ system_prd }}`

- **IMPLEMENTATION PLAN**: `{{ implementation_plan }}`


- **OUTPUT PATH**: `{{ output_dir }}`

## Goal

Translate the SCoT files under `{{ output_dir }}` into working source code, tests, configuration, scripts, and documentation under `{{ output_dir }}`.

The SCoT files are planning artifacts only. They should guide implementation, but they are not the final output.

## Required Behavior

You must:

1. Read `{{ output_dir }}/SCOT_INDEX.scot.md`.
2. Read every `*.scot.md` file referenced by the index.
3. Derive the final implementation file paths from the SCoT files.
4. Create the corresponding real implementation files under `{{ output_dir }}`.
5. Preserve the directory structure described by the SCoT artifacts.
6. Implement all required functionality from the PRD, implementation plan, and SCoT files.
7. Include tests, configuration, scripts, and documentation where the SCoT files call for them.
8. Write complete, functional, production-ready files.
9. Do not write new SCoT files.
10. Do not output code only in chat.

## File Translation Rule

Each SCoT file maps to one final implementation file.

Use this mapping:

```text
SCoT file:
  <path>/<filename>.<extension>.scot.md

Final implementation file:
  <path>/<filename>.<extension>
```

Examples:

```text
{{ output_dir }}/src/config.ts.scot.md
  -> {{ output_dir }}/src/config.ts

{{ output_dir }}/src/routes/health.ts.scot.md
  -> {{ output_dir }}/src/routes/health.ts

{{ output_dir }}/tests/health.test.ts.scot.md
  -> {{ output_dir }}/tests/health.test.ts

{{ output_dir }}/README.md.scot.md
  -> {{ output_dir }}/README.md
```

Do not copy the SCoT text into the final files. Use it as the blueprint for the actual implementation.

## Implementation Priorities

Follow this order unless the SCoT index specifies a more precise dependency order:

1. Project metadata and configuration files
2. Shared types, schemas, constants, and utilities
3. Environment/configuration loading
4. Data models, database clients, or persistence layers
5. Core domain logic and services
6. API routes, controllers, commands, UI components, or user-facing entry points
7. Application startup or orchestration files
8. Tests
9. Documentation

## Implementation Requirements

For each final file:

- Implement the responsibilities described in the matching SCoT file.
- Follow the sequence structure from the SCoT file.
- Implement all branch behavior described in the SCoT file.
- Implement all loop or repeated-workflow behavior described in the SCoT file.
- Respect dependencies and implementation order.
- Include appropriate error handling.
- Validate inputs where required.
- Handle edge cases identified in the SCoT file.
- Keep code readable, maintainable, and idiomatic for the project’s language and framework.
- Add comments only for complex or non-obvious logic.
- Add docstrings or JSDoc where useful for public functions, classes, modules, or APIs.
- Avoid overengineering beyond the PRD, plan, and SCoT artifacts.

## Testing Requirements

Create all test files described by the SCoT artifacts.

Tests must cover:

- Main success paths
- Input validation
- Error paths
- Edge cases
- Integration between components
- PRD acceptance criteria
- Branches identified in SCoT files
- Loops, batch behavior, retries, pagination, or repeated workflows where applicable

Tests should be deterministic and should avoid external network or service dependencies unless explicitly required.

Use mocks, fixtures, or test doubles where the SCoT files recommend them.

## Documentation Requirements

Create all documentation files described by the SCoT artifacts.

Documentation should include, where relevant:

- Setup instructions
- Installation instructions
- Environment variables
- Configuration options
- Usage examples
- Test commands
- Development workflow
- Operational notes
- API examples
- Known assumptions or limitations

## Handling Ambiguity

If a SCoT file marks something as ambiguous:

1. Use the safest reasonable implementation consistent with the PRD and implementation plan.
2. Document the assumption in the relevant code comments or documentation.
3. Do not block implementation unless the ambiguity makes the implementation impossible.

If there is a conflict between sources, resolve priority in this order:

1. PRD
2. Implementation plan
3. `SCOT_INDEX.scot.md`
4. Individual `*.scot.md` files
5. Reasonable engineering judgment

If a SCoT artifact is clearly technically flawed, make a pragmatic correction and document the deviation in the final summary.

## Output Rules

You must write files to:

```text
{{ output_dir }}
```

You must not write files outside:

```text
{{ output_dir }}
```

Do not write SCoT artifacts into the output path unless explicitly required as documentation.

Do not delete SCoT files. Do not overwrite unrelated existing files unless they are the exact final implementation targets derived from matching SCoT files.

Do not leave placeholder implementations such as:

```text
TODO
stub
not implemented
coming soon
placeholder
```

Do not omit required functionality.

Do not output only explanations. The primary output must be the actual files.

## Important Reminder

The SCoT files are not the final implementation.

They are blueprints.

Your job is to turn them into functional code, tests, configuration, scripts, and documentation.
