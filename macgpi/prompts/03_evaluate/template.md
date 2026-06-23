
# Software Implementation Evaluation Prompt

You are an expert software architect tasked with evaluating an implementation based on a system PRD and an implementation plan.

## Input
- **PRD**: {{ system_prd }}

- **PLAN**: {{ implementation_plan }}

- **Code**: All code located in the {{ output_dir }}

## Your Task
Provide a thorough, evidence-based evaluation of the implementation against the PRD and plan. Your evaluation should cover:

First, read the top-level `type` field from the implementation plan:

- `modification`: evaluate an existing project that has been changed to fix an issue or add an enhancement.
- `greenfield`: evaluate a new project built from the ground up.

Apply the common evaluation criteria below, then apply the type-specific criteria.

- Conformance: Does the implementation meet stated requirements?
- Architecture: Are architectural decisions correct and justified? Do they follow best practices?
- Components: Are module boundaries, interfaces and responsibilities correct?
- Dependencies: Are external/internal dependencies appropriate and justified?
- Quality & Testability: Code quality, test coverage targets, and test strategy. For Code Quality assess the ISO quality model found in ISO 25010:2011. Assess the following domains of the 25010:2011 quality model: Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability. Focus especially on the Maintainability of the artifact.
- Risks & Issues: Missing requirements, high-risk design choices, and security/privacy concerns. Follow common security pitfalls and kmnown attack methods and find vulnerabilities which should be fixed.  
- Actionable Recommendations: Prioritized fixes, owners, and verification steps.
- Security: Check for OWASP Top 10 patterns, hardcoded secrets, missing input validation, and insecure dependencies.

For `modification` plans, also evaluate:

- Minimality: Are changes limited to files and units needed for the requested fix or enhancement?
- Preservation: Does the implementation preserve existing public behavior, APIs, style, tests, and architecture unless the plan explicitly changes them?
- Regression coverage: Were tests added or updated to cover the changed behavior where feasible?
- Integration with existing code: Does the change follow existing project conventions and extension points?
- Diff risk: Are there broad rewrites, generated artifacts, documentation churn, dependency changes, or formatting-only changes unrelated to the PRD?
- Backward compatibility: Could existing consumers, tests, or documented behavior break unexpectedly?
- Validation fit: Were targeted checks or tests run or identified for the changed behavior?
- Artifact hygiene: Did SCoT files, generated planning files, or private implementation notes remain as final product output when they were not requested?

For `greenfield` plans, also evaluate:

- Completeness: Does the project include the source, tests, configuration, scripts, and documentation required by the PRD and plan?
- Runnability: Can the project be installed, started, and tested using the documented commands?
- Project shape: Is the initial structure coherent, idiomatic, and maintainable for the chosen architecture?
- Acceptance coverage: Do tests and documentation cover the primary workflows and acceptance criteria?
- Setup quality: Are dependencies, environment variables, and configuration documented and reproducible?
- Validation fit: Were project-level build/test/run checks performed or clearly documented?
- Artifact completeness: Are required project files present, including metadata, entry points, tests, and setup documentation?

Be concise but specific. Where possible, cite lines, files or test names from the implementation.

## Type-Specific Validation Instructions

Use `implementation_plan.type` to choose the evaluation emphasis and validation evidence.

For `modification`:

- Inspect the changed files and compare them to the PRD, implementation tasks, and SCoT preservation boundaries.
- Verify that the requested issue or enhancement is addressed without broad unrelated changes.
- Prefer targeted tests for the changed behavior. If tests were not run or cannot be run, state that in `test_results` notes and create a `test` finding when coverage is important.
- Treat unrelated rewrites, generated artifacts, dependency churn, formatting-only changes, or project-wide scaffolding as findings unless explicitly required by the PRD or plan.
- Treat leaked internal artifacts as findings unless the PRD explicitly requests them as product files. Internal artifacts include `SCOT_INDEX.scot.md`, `*.scot.md`, generated implementation plans, generated evaluation reports, private notes, and generated architecture/design documents not requested by the PRD.
- If internal artifacts leaked into final output, create a finding with `finding_type: "conformance"` or `"quality"` and a recommendation to remove them from the modification patch.
- Use `finding_type: "conformance"` for scope or preservation problems.
- Use `finding_type: "test"` for missing regression coverage or missing targeted validation.
- Use `finding_type: "quality"` for maintainability or diff-risk problems.

For `greenfield`:

- Inspect whether the project has the complete file set needed for a runnable first version.
- Check for project metadata, dependency declarations, source files, tests, documentation, configuration, scripts, and entry points where relevant.
- Prefer project-level build/test/run checks. If checks were not run or cannot be run, state that in `test_results` notes and create a finding if this leaves acceptance criteria unverified.
- Treat missing setup docs, missing tests, placeholder code, broken entry points, or inconsistent source/config/docs as findings.
- Use `finding_type: "conformance"` for missing requirements or incomplete workflows.
- Use `finding_type: "test"` for missing or failing project-level tests.
- Use `finding_type: "architecture"` or `"quality"` for incoherent structure, overengineering, or maintainability problems.

The schema does not allow extra top-level fields. Encode the type-specific assessment using existing fields:

- Put a concise type-specific verdict in `summary`.
- Put requirement-by-requirement status in `compliance`.
- Put concrete type-specific problems in `findings`.
- Put verification and remediation steps in `recommendations`.
- Put commands run, missing commands, or unverified checks in `test_results` notes.

When deciding the `next` phase in the evaluation JSON:

- Use `finish` only when the implementation satisfies the PRD, plan, and type-specific criteria well enough to be accepted.
- Use `revise` when there are missing requirements, failing or absent critical tests, unsafe broad changes, broken setup, or type-inappropriate output.
- For `modification`, choose `revise` if the implementation is correct in spirit but too broad, rewrites unrelated files, omits an important regression test, or changes behavior outside the requested scope.
- For `modification`, choose `revise` if SCoT files, private notes, or unrequested generated documentation remain as product output.
- For `modification`, choose `finish` only when the final output is focused, relevant, preserves existing behavior, and has adequate targeted validation or a clear explanation for unrun tests.
- For `greenfield`, choose `revise` if the project is incomplete, not runnable, missing essential tests/docs/configuration, or contains placeholder functionality.
- For `greenfield`, choose `finish` only when the project has a coherent runnable structure and the documented build/test/run path is present and plausible.

## Output Format
Provide the evaluation in structured json following the schema. ONLY generate the evaluation, do NOT generate anything else.
The following schema MUST be followed.
{{ schema_format }}

## Output location
**IMPORTANT:** ONLY WRITE YOUR RESULTS TO {{ output_file }}
