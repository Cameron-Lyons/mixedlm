# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in mixedlm, please report it by emailing the maintainers directly rather than opening a public issue.

When reporting a vulnerability, please include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Any suggested fixes (if available)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days, including an assessment of the vulnerability and planned remediation steps.

## Security Measures

This project implements the following security measures:

- **Dependency Scanning**: Automated weekly scans using pip-audit and cargo-audit
- **Code Analysis**: CodeQL static analysis on all pushes and pull requests
- **Security Linting**: Bandit security linter for Python code
- **Dependency Review**: Automatic review of dependency changes in pull requests
- **Pinned Action Versions**: All GitHub Actions use pinned versions to prevent supply chain attacks
- **Dependabot**: Automated dependency updates with security patches
