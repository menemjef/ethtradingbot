"""
audit_bridge/sanitizer.py — Comprehensive secret detection and redaction.

Recognizes and redacts:
- Bearer tokens & Authorization headers
- JSON Web Tokens (JWT)
- API keys & secrets (Binance, AWS, generic)
- Telegram bot tokens
- Passwords & credentials
- Private keys (RSA, EC, OpenSSH, DSA, PGP)
- Database connection strings (Postgres, MySQL, Mongo, Redis, SQLite)

Also provides publication safety gating: blocks publication if unredacted secrets are detected.
"""

import json
import re
from typing import Any, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# Regex Patterns for Sensitive Data
# ─────────────────────────────────────────────────────────────────────────────

# Private keys (multiline blocks)
PATTERN_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.MULTILINE,
)

# Authorization header with Bearer or Basic token
PATTERN_AUTH_HEADER = re.compile(
    r"(?i)\bAuthorization\s*:\s*(?:Bearer|Basic|Token)\s+([A-Za-z0-9_\-\.\~+/=]{8,})",
)

# Standalone Bearer token
PATTERN_BEARER = re.compile(
    r"(?i)\bBearer\s+([A-Za-z0-9_\-\.\~+/=]{8,})",
)

# JSON Web Tokens (header.payload.signature)
PATTERN_JWT = re.compile(
    r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-\.\~+/=]{10,}\b",
)

# Telegram Bot Token (e.g. <bot_id>:<secret_token>)
PATTERN_TELEGRAM = re.compile(
    r"\b\d{8,12}:[A-Za-z0-9_-]{30,45}\b",
)

# Database Connection URIs (with embedded user:password)
PATTERN_DATABASE_URI = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss):\/\/[^\s:@\"']+:[^\s@\"']+@[^\s\"']+",
)

# Key-Value assignments for secrets, credentials, API keys, passwords
PATTERN_KEY_VALUE_SECRET = re.compile(
    r"(?i)\b("
    r"BINANCE_API_KEY|BINANCE_API_SECRET|API_KEY|API_SECRET|SECRET_KEY|"
    r"SECRET|PASSWORD|PASSWD|PWD|TELEGRAM_TOKEN|AUTH_TOKEN|ACCESS_TOKEN|"
    r"PRIVATE_KEY|CLIENT_SECRET|PRIVATE_TOKEN|ENCRYPTION_KEY"
    r")\s*([:=])\s*([\"']?)([A-Za-z0-9_\-\.\~+/=:\$]{6,})\3",
)

# JSON key-value secret pairs: "apiKey": "xyz..."
PATTERN_JSON_SECRET = re.compile(
    r'(?i)"([^"]*(?:key|secret|password|passwd|token|credential|auth)[^"]*)"\s*:\s*"([A-Za-z0-9_\-\.\~+/=]{6,})"',
)


def sanitize_text(text: str) -> str:
    """
    Sanitize all known secret patterns from a string, replacing them with [REDACTED].
    Preserves document structure while masking sensitive values.
    """
    if not text or not isinstance(text, str):
        return text

    sanitized = text

    # 1. Private keys (full block replacement)
    sanitized = PATTERN_PRIVATE_KEY.sub("[REDACTED_PRIVATE_KEY]", sanitized)

    # 2. Database connection strings (replace password part or full URI)
    def _redact_db_uri(m):
        full = m.group(0)
        # replace :password@ with :[REDACTED]@
        return re.sub(r":([^@:]+)@", ":[REDACTED]@", full)
    sanitized = PATTERN_DATABASE_URI.sub(_redact_db_uri, sanitized)

    # 3. JWTs
    sanitized = PATTERN_JWT.sub("[REDACTED_JWT]", sanitized)

    # 4. Telegram bot tokens
    sanitized = PATTERN_TELEGRAM.sub("[REDACTED_TELEGRAM_TOKEN]", sanitized)

    # 5. Authorization headers
    sanitized = PATTERN_AUTH_HEADER.sub(lambda m: m.group(0).split()[0] + " " + m.group(0).split()[1] + " [REDACTED]", sanitized)

    # 6. Standalone Bearer tokens
    sanitized = PATTERN_BEARER.sub("Bearer [REDACTED]", sanitized)

    # 7. Key-value secrets (e.g. BINANCE_API_KEY=xxx or API_KEY: "xxx")
    sanitized = PATTERN_KEY_VALUE_SECRET.sub(r"\1\2\3[REDACTED]\3", sanitized)

    # 8. JSON secret fields (e.g. "apiKey": "secret123")
    sanitized = PATTERN_JSON_SECRET.sub(r'"\1": "[REDACTED]"', sanitized)

    return sanitized


def sanitize_data(obj: Any) -> Any:
    """Recursively sanitize strings inside dictionaries, lists, or primitive types."""
    if isinstance(obj, str):
        return sanitize_text(obj)
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # If the key itself indicates a secret, redact value directly
            if any(s in k.lower() for s in ["api_key", "secret", "password", "token", "credential", "private_key"]):
                new_dict[k] = "[REDACTED]"
            else:
                new_dict[k] = sanitize_data(v)
        return new_dict
    elif isinstance(obj, list):
        return [sanitize_data(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_data(item) for item in obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Publication Safety Gating
# ─────────────────────────────────────────────────────────────────────────────

class PublicationSafetyError(Exception):
    """Raised when an artifact contains un-redacted secrets and cannot be safely published."""
    pass


def scan_for_unredacted_secrets(text: str) -> List[Tuple[str, str]]:
    """
    Scan text for any remaining un-redacted secrets.
    Returns a list of (category, matched_sample).
    """
    if not text or not isinstance(text, str):
        return []

    findings = []

    # Check for private keys
    if PATTERN_PRIVATE_KEY.search(text):
        findings.append(("PRIVATE_KEY", "Found un-redacted private key block"))

    # Check for JWT
    jwt_match = PATTERN_JWT.search(text)
    if jwt_match:
        findings.append(("JWT", jwt_match.group(0)[:20] + "..."))

    # Check for Telegram tokens
    tg_match = PATTERN_TELEGRAM.search(text)
    if tg_match:
        findings.append(("TELEGRAM_TOKEN", tg_match.group(0)[:15] + "..."))

    # Check for DB credentials
    db_match = PATTERN_DATABASE_URI.search(text)
    if db_match:
        findings.append(("DATABASE_URI", db_match.group(0)[:25] + "..."))

    # Check for Bearer token not redacted
    for m in PATTERN_BEARER.finditer(text):
        val = m.group(1)
        if val != "[REDACTED]":
            findings.append(("BEARER_TOKEN", f"Bearer {val[:10]}..."))

    # Check key-value secrets
    for m in PATTERN_KEY_VALUE_SECRET.finditer(text):
        val = m.group(4)
        if val != "[REDACTED]":
            findings.append((m.group(1), f"{val[:6]}..."))

    return findings


def assert_safe_for_publication(data: Any, context_name: str = "artifact") -> bool:
    """
    Verify that an artifact is completely free of un-redacted secrets before publication.
    Raises PublicationSafetyError if any un-redacted secrets are found.
    """
    if isinstance(data, (dict, list)):
        text = json.dumps(data)
    else:
        text = str(data)

    findings = scan_for_unredacted_secrets(text)
    if findings:
        reasons = "; ".join(f"{cat}: {sample}" for cat, sample in findings)
        raise PublicationSafetyError(
            f"Publication blocked: {context_name} contains unredacted secrets ({reasons})"
        )

    return True
