
# Software Implementation Planning Prompt

You are an expert software architect tasked with creating a detailed implementation plan.

## Input
- **PRD**: {{ system_prd }}

## Your Task

Analyze the provided PRD and generate a structured implementation plan that includes:

### 1. **Project Overview**
- Summary of key features and objectives
- Success criteria

### 2. **Architecture Design**
- High-level system architecture
- Technology stack recommendations
- Component breakdown

### 3. **Implementation Phases**
- Phase name and description
- Duration estimate
- Key deliverables
- Dependencies

### 4. **Technical Specifications**
- Data models and schemas
- API endpoints (if applicable)
- Database requirements

### 5. **Risk Assessment**
- Potential risks
- Mitigation strategies

### 6. **Resource Requirements**
- Team composition
- Tools and infrastructure

## Output Format
Provide the plan in structured json with clear sections and bullet points. ONLY generate the plan, do NOT generate any code yet.
The following schema MUST be followed.
{{ schema_format }}

## Output location
**IMPORTANT:** ONLY WRITE YOUR RESULTS TO {{ output_path }}