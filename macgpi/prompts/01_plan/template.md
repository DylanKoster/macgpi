
# Software Implementation Planning Prompt

You are an expert software architect tasked with creating a detailed implementation plan.

## Input

<prd>
{{ system_prd }}
</prd>

## Your Task

Analyze the provided PRD and produce a structured implementation plan with the following sections. Each section maps directly to the required output schema.

### 1. **Type**
- Either `modification` or `greenfield`
- Shows what kind of implementation task this is.
- Value: `modification` when there is an existing project that needs to be modified to fix an issue on add an enhancement.
- Value: `greenfield` when the project is completely new and is expected to be built from the ground up.  
- If unsure, inspect the project directory. Prefer `modification` when meaningful source files, tests, or project metadata already exist.

### 2. **Objectives**
- One entry per primary goal or requirement derived from the PRD
- Assign each an `id`, a clear `description`, and a `priority` (high / medium / low)
- Optionally include measurable `metrics` for each objective

### 3. **Architecture**
- Choose an architectural `pattern` (e.g. layered, MVC, microservices, event-driven)
- Provide a `rationale` explaining why this pattern suits the PRD
- List any `constraints` that restrict the design space

### 4. **Components**
- One entry per planned module or service
- State the `name` and its `responsibility`
- Optionally describe `interfaces` it exposes and a note on `testability`

### 5. **Dependencies**
- One entry per external library or internal component dependency
- State `name`, `type` (external / internal), and a `justification`
- Include a `version` where known

### 6. **Implementation Tasks**
- One entry per concrete unit of work, ordered by execution dependency
- Assign each an `id` and a short `title`
- Set `component` to the name of the component (from section 3) the task belongs to
- Write a `description` explaining exactly what must be built or configured
- List `depends_on` as the `id`s of any tasks that must complete first (omit if none)
- List `acceptance_criteria` as verifiable conditions that confirm the task is done

The tasks depend on the type of the implementation:

For `modification` plans, include in the relevant implementation tasks:
- Existing files or modules likely to change.
- Existing behavior that must be preserved.
- Targeted tests to add or update.
- Non-goals: unrelated files, broad rewrites, dependency changes, or documentation churn that should be avoided.

For `greenfield` plans, include in the relevant implementation tasks:
- Project structure to create.
- Source modules, tests, configuration, scripts, and documentation.
- Setup and run commands.
- Acceptance criteria for a runnable first version.

### 7. **Quality Standards**
- List the `coding_standards` to be enforced (e.g. linting rules, style guides)
- Set the `documentation_level` (minimal / standard / comprehensive)
- Set a `test_coverage_target` as a percentage (0–100)


## Output Format
Provide the plan in structured json with clear sections and bullet points. ONLY generate the plan, do NOT generate any code yet.
The following schema MUST be followed.
{{ schema_format }}

## Output location
**IMPORTANT:** ONLY WRITE YOUR RESULTS TO {{ output_file }}