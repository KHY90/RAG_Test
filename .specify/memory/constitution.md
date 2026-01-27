<!--
SYNC IMPACT REPORT
==================
Version change: 0.0.0 → 1.0.0
Bump rationale: Initial constitution creation (MAJOR - new governance structure)

Modified principles: N/A (initial creation)
Added sections:
  - Core Principles (3 principles: Retrieval Quality First, Test-First Development, Simplicity & Maintainability)
  - Development Workflow
  - Quality Gates
  - Governance

Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section exists)
  - .specify/templates/spec-template.md: ✅ Compatible (Requirements/Success Criteria sections exist)
  - .specify/templates/tasks-template.md: ✅ Compatible (Phase structure supports TDD)

Follow-up TODOs: None
-->

# RAGTest Constitution

## Core Principles

### I. Retrieval Quality First

Every feature MUST prioritize retrieval accuracy and relevance over other concerns.

- Retrieval pipelines MUST be measurable with precision/recall metrics
- Vector embeddings and chunking strategies MUST be documented and justified
- Context window utilization MUST be optimized for the target LLM
- Hallucination mitigation strategies MUST be implemented for all user-facing outputs
- Source attribution MUST be maintained throughout the retrieval-to-generation pipeline

**Rationale**: In RAG systems, the quality of retrieved context directly determines output quality.
Poor retrieval leads to hallucinations, irrelevant answers, and user distrust.

### II. Test-First Development (NON-NEGOTIABLE)

All feature development MUST follow strict Test-Driven Development.

- Tests MUST be written before implementation code
- Tests MUST fail before implementation begins (Red phase)
- Implementation MUST make tests pass with minimal code (Green phase)
- Refactoring MUST occur only after tests pass (Refactor phase)
- Retrieval quality tests MUST include:
  - Precision/recall benchmarks on representative queries
  - Edge case queries (ambiguous, multi-hop, out-of-scope)
  - Latency thresholds for retrieval operations

**Rationale**: TDD ensures correctness, enables safe refactoring, and documents expected behavior.
For RAG systems, test-first is critical to catch retrieval regressions early.

### III. Simplicity & Maintainability

Start simple. Add complexity only when proven necessary.

- YAGNI (You Aren't Gonna Need It) principle MUST guide all decisions
- New abstractions MUST be justified by concrete, current requirements
- Dependencies MUST be minimized; each dependency MUST solve a real problem
- Configuration MUST have sensible defaults; avoid premature parameterization
- Code MUST be readable without extensive comments; self-documenting names preferred

**Rationale**: RAG systems can become complex quickly with embeddings, vector stores, chunking
strategies, and LLM integrations. Simplicity reduces bugs and maintenance burden.

## Development Workflow

- Feature branches MUST be created from main for all changes
- Commits MUST be atomic and focused on a single logical change
- Pull requests MUST include:
  - Description of the change and its purpose
  - Test coverage for new functionality
  - Retrieval quality impact assessment (if applicable)
- Code review MUST verify compliance with all Constitution principles

## Quality Gates

All changes MUST pass these gates before merge:

1. **Test Suite**: All tests MUST pass (unit, integration, retrieval benchmarks)
2. **Retrieval Metrics**: Precision/recall MUST not regress below baseline
3. **Latency Budget**: Retrieval operations MUST complete within defined SLAs
4. **Code Review**: At least one approval from a team member
5. **Constitution Compliance**: Reviewer MUST verify adherence to these principles

## Governance

This Constitution is the authoritative source for project standards and practices.

**Amendment Process**:
1. Propose amendment via pull request to this file
2. Document rationale and impact on existing code
3. Obtain team consensus before merge
4. Update version according to semantic versioning:
   - MAJOR: Principle removal or incompatible redefinition
   - MINOR: New principle or significant guidance expansion
   - PATCH: Clarifications, typos, non-semantic refinements
5. Propagate changes to dependent templates and documentation

**Compliance**:
- All PRs and code reviews MUST verify Constitution compliance
- Violations MUST be documented and justified in Complexity Tracking (plan.md)
- Unjustified violations MUST block merge

**Version**: 1.0.0 | **Ratified**: 2026-01-27 | **Last Amended**: 2026-01-27
