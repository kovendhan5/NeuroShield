# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly by emailing the maintainers. Do not open a public issue.

## Security Measures

### Secrets Management
- **Never commit `.env` files** — they are in `.gitignore`
- Use `.env.example` as a template (contains no real credentials)
- Jenkins credentials use API tokens, not account passwords
- Email uses Gmail App Passwords, not account passwords
- All secrets are loaded from environment variables at runtime

### API Security
- **CORS** is restricted to explicit origins (configurable via `CORS_ALLOWED_ORIGINS`)
- OpenAPI docs (`/docs`, `/redoc`) are disabled in production (`NEUROSHIELD_ENV=production`)
- Healing action trigger endpoint validates action names against an allowlist
- All subprocess calls use argument lists (no `shell=True`) to prevent command injection

### Input Validation
- Kubernetes namespace and service names are validated against the K8s naming regex
- Scale replica counts are validated as positive integers
- Email subjects use `email.header.Header()` to prevent header injection
- Slack webhook URLs are validated to only allow `https://*.slack.com`

### Model Loading
- `torch.load()` uses `weights_only=True` to prevent arbitrary code execution
- `joblib.load()` is used only for trusted PCA models from our own training pipeline
- Model files (`.pth`, `.joblib`) are in `.gitignore`

### Infrastructure
- Docker Compose uses named volumes (not host-path bind mounts)
- Prometheus config is mounted read-only
- Health checks are configured for all services
- Dummy app destructive endpoints (`/crash`, `/stress`, `/fail`) require API key authentication

### Telemetry
- Sensitive data (API keys, tokens, passwords, bearer tokens) is redacted from Jenkins logs before storage
- CSV data files are in `.gitignore`

## Checklist for Contributors

- [ ] No hardcoded secrets or credentials
- [ ] Environment variables for all configurable values
- [ ] Input validation at system boundaries
- [ ] No `shell=True` in subprocess calls
- [ ] Error messages don't leak sensitive information
- [ ] Model files loaded with safe options
