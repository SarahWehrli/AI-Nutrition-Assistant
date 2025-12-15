"""
Security utilities for SSRF protection and safe HTTP requests.

Provides safe HTTP GET functionality with host allowlisting,
private IP blocking, redirect prevention, and response size limits.
"""

import socket
import ipaddress
import requests
from urllib.parse import urlparse

# ============================================================================
# Configuration
# ============================================================================

# Only hosts your app is allowed to contact via HTTP
ALLOWED_HTTP_HOSTS = {
    "foodish-api.com",
    "picsum.photos",
}

# ============================================================================
# Helper Functions
# ============================================================================

def _is_public_ip(ip_str: str) -> bool:
    """Check if an IP address is public (not private, loopback, etc.)."""
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )

# ============================================================================
# Public API
# ============================================================================
def safe_get(url: str, *, timeout=5, allow_redirects=False, max_bytes=1_000_000):
    """
    SSRF-safe HTTP GET:
    - allowlist hosts
    - block localhost / private IPs
    - block redirects
    - cap response size
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked URL scheme: {parsed.scheme}")

    host = parsed.hostname
    if not host:
        raise ValueError("Blocked URL without hostname")

    host = host.lower()
    if host not in ALLOWED_HTTP_HOSTS:
        raise ValueError(f"Blocked host: {host}")

    # DNS resolution + private IP protection
    infos = socket.getaddrinfo(
        host,
        parsed.port or (443 if parsed.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )
    resolved_ips = {info[4][0] for info in infos}
    if any(not _is_public_ip(ip) for ip in resolved_ips):
        raise ValueError(f"Blocked non-public IP for host {host}: {resolved_ips}")

    r = requests.get(
        url,
        timeout=timeout,
        allow_redirects=allow_redirects,
        stream=True,
    )
    r.raise_for_status()

    if 300 <= r.status_code < 400:
        raise ValueError("Redirects are not allowed")

    # Limit response size
    data = bytearray()
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError("Response too large")

    r._content = bytes(data)
    return r