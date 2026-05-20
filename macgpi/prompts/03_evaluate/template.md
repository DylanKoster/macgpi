
# Software Implementation Evaluation Prompt

You are an expert software architect tasked with evaluating an implementation based on a system PRD and an implementation plan.

## Input
- **PRD**: {{ system_prd }}

- **PLAN**: {{ implementation_plan }}

- **Code**: All code located in the {{ output_dir }}

## Your Task
Provide a thorough, evidence-based evaluation of the implementation against the PRD and plan. Your evaluation should cover:

- Conformance: Does the implementation meet stated requirements?
- Architecture: Are architectural decisions correct and justified? Do they follow best practices?
- Components: Are module boundaries, interfaces and responsibilities correct?
- Dependencies: Are external/internal dependencies appropriate and justified?
- Quality & Testability: Code quality, test coverage targets, and test strategy. For Code Quality assess the ISO quality model found in ISO 25010:2011. Assess the following domains of the 25010:2011 quality model: Functional Suitability, Performance Efficiency, Compatibility, Usability, Reliability, Security, Maintainability, Portability. Focus especially on the Maintainability of the artifact.
- Risks & Issues: Missing requirements, high-risk design choices, and security/privacy concerns. Follow common security pitfalls and kmnown attack methods and find vulnerabilities which should be fixed.  
- Actionable Recommendations: Prioritized fixes, owners, and verification steps.
- Security: Check for OWASP Top 10 patterns, hardcoded secrets, missing input validation, and insecure dependencies.

Be concise but specific. Where possible, cite lines, files or test names from the implementation.

## Output Format
Provide the evaluation in structured json following the schema. ONLY generate the evaluation, do NOT generate anything else.
The following schema MUST be followed.
{{ schema_format }}

## Output location
**IMPORTANT:** ONLY WRITE YOUR RESULTS TO {{ output_file }}