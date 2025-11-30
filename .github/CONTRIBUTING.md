# Contribution Guidelines

This repository follows a 3-person team workflow with branch-based development and PR review.

Branching
- `main`: protected, only reviewed PRs merged
- feature branches: `feature/<short-desc>`
- experiment branches: `exp/<short-desc>`

Pull Requests
- Open a PR from your feature branch to `main` (or `develop` if used).
- Include description, data/sample used, and training results.
- At least one teammate must approve before merge.

Testing and Code Style
- Add unit tests to `tests/` for non-trivial functions.
- Use `pytest` for test runs.

Commit messages
- Use concise, imperative messages: `Add data split script`.
