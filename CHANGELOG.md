# Changelog

All notable changes to BenefitIQ. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-19 - Sprint 1: Foundation

### Added
- Project skeleton, pinned tooling, pytest + ruff config and GitHub Actions CI (BIQ-1, #17)
- Typed settings from `.env` with validation and `SecretStr` keys; `.env.example` (BIQ-2, #18)
- Claude on Amazon Bedrock LLM layer: Sonnet 5 via Converse with Haiku 4.5 fallback, Titan Text Embeddings V2,
  offline hash embeddings, botocore timeouts and adaptive retries, `scripts/check_bedrock.py` (BIQ-3, #19)
- PII/PHI masking guardrail with reversible placeholders and member-ID extraction (BIQ-4, #20)
- Prompt-injection guardrail with seven named rules and a pluggable classifier hook (BIQ-5, #21)

### Fixed
- SQL injection rule no longer blocks plain English such as "select the best plan from the list" (#21)

[Unreleased]: https://github.com/sai-4321/benifitiq/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sai-4321/benifitiq/releases/tag/v0.1.0
