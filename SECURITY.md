# Security

## Report a vulnerability

Do not open a public issue.
Use GitHub private vulnerability reporting: open the Security tab of
gildrb/elpis-fast and select "Report a vulnerability".
You can also send the report to mail@gildrb.com.
Include the affected files, the steps to reproduce, and the impact.

## Scope

This policy covers the elpis-fast repository.

- `serve/`: the authenticated HTTP API and its tool-call validation.
- `Dockerfile.exl3`, `docker/`, `docker-compose.yml`: image builds and the launch recipe.
- `patches/`, `bend/`: engine changes and the acceptance code that the image runs.
- `eval/`, `bench/`: tools that download and run third-party code.
- `flake.nix`, `nix/`: pinned toolchains and deployment adapters.

## Supported versions

Only `main` gets fixes.
