# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Yes     |
| 1.x     | ❌ No      |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a vulnerability in PlantDx, report it privately via GitHub's
built-in security advisory system:

1. Go to the [Security tab](https://github.com/Pelex04/Diagnosis-plant-model/security)
2. Click **"Report a vulnerability"**
3. Fill in the details

You will receive a response within **72 hours** acknowledging the report.
We aim to release a patch within **14 days** for confirmed vulnerabilities.

## Scope

Security issues relevant to this project include:

- **Arbitrary code execution** via malicious model checkpoint files (`.pth`)
- **Path traversal** in batch inference file handling
- **Dependency vulnerabilities** in pinned packages (`requirements.txt`)

## Out of Scope

- Bugs that do not have security implications
- Issues in PyTorch, torchvision, or other upstream dependencies
  (report these to the respective maintainers)

## Disclosure Policy

Once a fix is released, we will publish a GitHub Security Advisory describing
the vulnerability, its impact, and the remediation.
