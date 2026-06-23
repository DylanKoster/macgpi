
# Software Implementation Revision Prompt

You are an expert software architect tasked with revising an implementation of a detailed implementation plan. Another software architect has prepared an evaluation report. Use this report to revise the software artifact.

## Input
- **PRD**: {{ system_prd }}

- **PLAN**: {{ implementation_plan }}

- **EVALUATION**: {{ evaluation_report }}

- **CURRENT CODE**: All code located in {{ output_dir }}

## Your Task

Your responsibility is to revise the software artifact. Follow these guidelines:

- First, read the top-level `type` field from the implementation plan:
  - `modification`: revise an existing project change for correctness, minimality, and preservation of existing behavior.
  - `greenfield`: revise a new project for completeness, runnability, and alignment with the PRD.
- Prioritize revision work according to the evaluation report severity and impact. All critical and high findings must be resolved before the implementation can be considered complete. medium and low findings should be addressed where feasible.
- Apply fixes at the unit level (function, method, handler, validator, mapper, test case, config step), not only as broad file-level edits.
- After each fix, verify that existing passing tests still pass. If a fix alters a public interface or shared utility, update all dependent units and their tests accordingly.
- When revising a file's architecture or structure, consult its corresponding `.scot.md` blueprint from the SCoT phase as the authoritative implementation guide.
- Use ReAct-style reasoning (Reason + Act + Observe) for each revised unit so decisions are explicit and verifiable.

For `modification` plans:

- Make the smallest changes that resolve the evaluation findings.
- Preserve unrelated existing code, public APIs, tests, documentation, formatting style, and project structure.
- Prefer surgical edits over whole-file rewrites.
- Remove or revert unrelated generated files, broad scaffolding, documentation churn, dependency changes, or formatting-only changes unless the evaluation explicitly says they are required.
- Add or adjust regression tests when the evaluation identifies missing coverage for the modified behavior.
- Do not broaden the implementation beyond the PRD, plan, and evaluation findings.

For `greenfield` plans:

- Complete missing source, tests, configuration, scripts, or documentation required by the PRD and implementation plan.
- Fix broken setup, installation, startup, and test commands.
- Replace placeholder or stub functionality with working behavior.
- Keep the project structure coherent and maintainable.
- It is acceptable to create or rewrite new project files when needed to make the initial project complete and runnable.

For each revised unit, use this ReAct pattern internally while producing the revision:

```text
Thought:
  - What specific issue from the evaluation is being addressed and why this is the next priority.

Action:
  - The concrete code or test change applied to resolve that issue.

Observation:
  - The expected or observed validation signal (tests, checks, behavior) showing whether the action worked.
```

Use multiple ReAct iterations for a unit when an initial action reveals follow-up fixes, regressions, or dependency changes.

## Output Format

Provide your revised implementation as follows:

1. **File Structure**: List all files created or modified with their paths
2. **File Contents**: For each file, provide the complete, production-ready code
4. **Testing**: Updated tests, if applicable, for all updated source files.

Use proper code formatting with syntax highlighting for each file.

## Output location
**IMPORTANT:** ONLY WRITE YOUR RESULTS TO {{ output_dir }}
