# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| < 1.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in mixedlm, please report it privately through
[GitHub's vulnerability reporting form](https://github.com/Cameron-Lyons/mixedlm/security/advisories/new)
rather than opening a public issue.

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Any suggested fixes (if available)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days, including an assessment of the vulnerability and planned remediation steps.

## Security Measures

This project implements the following security measures:

- **Dependency Scanning**: pip-audit checks the locked Python runtime dependencies, including optional extras, and cargo-audit checks both Rust lockfiles
- **Security Linting**: Bandit scans the Python sources
- **Dependency Review**: Pull requests that add dependencies with high-severity advisories or disallowed licenses fail review
- **Code Analysis**: CodeQL analyzes the workflows, Python, and Rust code on pull requests, pushes to `main`, and weekly
- **Dependabot**: Weekly update pull requests for `uv.lock`, the Cargo lockfiles, and GitHub Actions

The scans, linting, and dependency review run in every pull request's required CI checks, and the Security workflow repeats the scans weekly. GitHub Actions are referenced by version tags rather than commit SHAs, and Dependabot updates them.
