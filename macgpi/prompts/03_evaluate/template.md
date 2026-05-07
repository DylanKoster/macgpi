
# Software Implementation Evaluation Prompt

You are an expert software architect tasked with evaluating an implementation based on a system PRD and an implementation plan.

## Input
- **PRD**: {{ system_prd }}

- **PLAN**: {{ implementation_plan }}

- **Code**: All code located in the {{ output_dir }}

## Your Task
Provide a thorough, evidence-based evaluation of the implementation against the PRD and plan. Your evaluation should cover:

- Conformance: Does the implementation meet stated requirements?
- Architecture: Are architectural decisions correct and justified?
- Components: Are module boundaries, interfaces and responsibilities correct?
- Dependencies: Are external/internal dependencies appropriate and justified?
- Quality & Testability: Code quality, test coverage targets, and test strategy.
- Risks & Issues: Missing requirements, high-risk design choices, and security/privacy concerns.
- Metrics: Measurable indicators and acceptance criteria.
- Actionable Recommendations: Prioritized fixes, owners, and verification steps.

Be concise but specific. Where possible, cite lines, files or test names from the implementation.

## Output Format
Provide the evaluation in structured json with clear sections and bullet points. ONLY generate the plan, do NOT generate anything else.
The following schema MUST be followed.
{{ schema_format }}

## Output location
**IMPORTANT:** ONLY WRITE YOUR RESULTS TO {{ output_path }}