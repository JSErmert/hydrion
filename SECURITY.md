# Security Policy

## Supported Versions

HydrOS is research / portfolio software under active development. Only the `main` branch is considered current. Older branches and tags are not maintained.

| Branch | Supported |
| ------ | --------- |
| `main` | Yes       |
| Others | No        |

## Reporting a Vulnerability

If you discover a security issue in HydrOS — whether in the Python simulation engine, the FastAPI service layer, or the WebGL/React console — please report it privately rather than opening a public issue.

**Contact:** [jseermert@gmail.com](mailto:jseermert@gmail.com)

Please include:

- A clear description of the issue
- Steps to reproduce
- The affected file path(s) or endpoint(s)
- Any proof-of-concept code or output (if applicable)

I aim to acknowledge within 5 business days. Coordinated disclosure is appreciated — please give me a reasonable window to investigate and patch before any public discussion.

## Scope

In scope:
- The Python engine under `hydrion/`
- The FastAPI service in `hydrion/service/`
- The frontend in `apps/hydros-console/`
- Repository configuration and CI

Out of scope:
- Vulnerabilities in third-party dependencies (please report upstream)
- Issues that require physical access to a deployed hardware variant (HydrOS does not yet have a hardware deployment)
- Denial-of-service via heavy simulation workloads (resource-bound by design)
