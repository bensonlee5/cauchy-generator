# Security Policy

## Supported versions

Security fixes are targeted at the current release line.

| Version    | Supported |
| ---------- | --------- |
| `0.19.x`   | yes       |
| `< 0.19.0` | no        |

## Reporting a vulnerability

Please do not open a public issue for an undisclosed security vulnerability.

Preferred path:

1. Use GitHub Security Advisories private reporting for this repository.
1. If you cannot access private reporting, open a minimal public issue only for
   non-sensitive hardening questions, not for active exploit details.

When reporting, include:

- affected version
- impact summary
- reproduction steps or proof of concept
- whether the issue affects generated artifact integrity, packaged recipes, or
  the CLI/runtime surface

## Scope notes

The main security-sensitive surfaces in `dagzoo` are:

- packaged CLI and config loading
- generated artifact integrity and metadata contracts
- CI/release automation
- dependency supply chain and published packages
