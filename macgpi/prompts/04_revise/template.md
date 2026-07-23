
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
- Revision must fix both functional defects and type-inappropriate output. A technically plausible change is not acceptable if it violates the implementation plan type.

For `modification` plans:

- Make the smallest changes that resolve the evaluation findings.
- Preserve unrelated existing code, public APIs, tests, documentation, formatting style, and project structure.
- Prefer surgical edits over whole-file rewrites.
- Remove or revert unrelated generated files, broad scaffolding, documentation churn, dependency changes, or formatting-only changes unless the evaluation explicitly says they are required.
- Add or adjust regression tests when the evaluation identifies missing coverage for the modified behavior.
- Do not broaden the implementation beyond the PRD, plan, and evaluation findings.
- If the evaluation says the patch is too broad, reduce it to the smallest source/test/doc changes needed for the requested issue or enhancement.
- If implementation created replacement modules parallel to existing modules, remove the replacement unless explicitly required and move the needed behavior into the existing extension point.
- If implementation rewrote an existing file wholesale, restore unrelated sections and keep only the necessary changed units.
- If SCoT files, planning files, private notes, or generated architecture documents leaked into the product output, remove them unless the PRD explicitly requests them.
- If dependencies, project metadata, or formatting were changed without clear need, restore them.
- If a regression test is missing, add or update the narrowest relevant test.

For `greenfield` plans:

- Complete missing source, tests, configuration, scripts, or documentation required by the PRD and implementation plan.
- Fix broken setup, installation, startup, and test commands.
- Replace placeholder or stub functionality with working behavior.
- Keep the project structure coherent and maintainable.
- It is acceptable to create or rewrite new project files when needed to make the initial project complete and runnable.
- If evaluation says the project is incomplete, add the missing runnable-project pieces rather than only documenting the gap.
- If source, tests, configuration, documentation, and scripts disagree, update them to describe and exercise the same project behavior.
- If dependency declarations or entry points are missing, add them when needed for the project to install, run, or test.
- If tests are missing, add tests for core workflows and acceptance criteria.
- If setup documentation is missing or wrong, add or correct install, run, and test instructions.

## Type-Inappropriate Output Cleanup

Use this cleanup checklist before finishing revision.

For `modification`:

- Remove files that are only planning artifacts, including SCoT-only artifacts, unless explicitly requested as product documentation.
- Remove `SCOT_INDEX.scot.md` and `*.scot.md` files from product output unless explicitly requested.
- Remove generated implementation plans, generated evaluation reports, private notes, and generated architecture/design documents that are not requested by the PRD.
- Remove broad generated scaffolding unrelated to the requested change.
- Remove unrelated docs/config/dependency changes.
- Restore unrelated code that was reformatted, reorganized, renamed, or moved.
- Keep the final change focused on the target behavior and its tests.

For `greenfield`:

- Do not remove required scaffolding just because it is broad; a new project needs a complete structure.
- Remove placeholder files that do not contribute to a runnable project.
- Ensure every created file has a clear purpose in the project.
- Ensure the README or setup docs match the actual files and commands.

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

When describing each revised unit internally, include whether the change is:

- a functional correction
- a scope reduction
- artifact cleanup
- test coverage repair
- greenfield completeness repair

## Output Format

Provide your revised implementation as follows:

1. **File Structure**: List all files created or modified with their paths
2. **File Contents**: For each file, provide the complete, production-ready code
3. **Removed Files**: For `modification` revisions, list any unrelated generated, planning, or broad-scaffolding files removed.
4. **Testing**: Updated tests, if applicable, for all updated source files.
5. **Scope Note**: State how the revision respects `modification` minimality or `greenfield` completeness.

Use proper code formatting with syntax highlighting for each file.

## Output location
**IMPORTANT:** ONLY WRITE YOUR RESULTS TO {{ output_dir }}
