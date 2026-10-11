"""
AWE Security & Vulnerability Scanner
======================================
Analyzes websites for build errors, broken links, SSL issues,
security headers, mixed content, and exposed secrets.
"""

import asyncio
import re
import ssl
import socket
import os
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")


# =============================================================================
# Severity Levels
# =============================================================================

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"
PASS = "pass"


# =============================================================================
# Secret Patterns
# =============================================================================

SECRET_PATTERNS = [
    (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})', "API Key"),
    (r'(?i)(secret|password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']{8,})', "Password/Secret"),
    (r'(?i)(token|auth[_-]?token|access[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{20,})', "Auth Token"),
    (r'(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*["\']?(AKIA[A-Z0-9]{16})', "AWS Access Key"),
    (r'(?i)(private[_-]?key)\s*[:=]\s*["\']?([^\s"\']{20,})', "Private Key Reference"),
    (r'sk-[a-zA-Z0-9]{32,}', "OpenAI API Key"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
    (r'gsk_[a-zA-Z0-9]{20,}', "Groq API Key"),
]

# Known library version patterns
LIBRARY_PATTERNS = {
    "jquery": r'jquery[.-]?v?(\d+\.\d+\.?\d*)',
    "bootstrap": r'bootstrap[.-]?v?(\d+\.\d+\.?\d*)',
    "angular": r'angular[.-]?v?(\d+\.\d+\.?\d*)',
    "react": r'react[.-]?v?(\d+\.\d+\.?\d*)',
    "vue": r'vue[.-]?v?(\d+\.\d+\.?\d*)',
    "lodash": r'lodash[.-]?v?(\d+\.\d+\.?\d*)',
    "moment": r'moment[.-]?v?(\d+\.\d+\.?\d*)',
}


# =============================================================================
# Core Scanner Functions
# =============================================================================

async def check_ssl(url: str) -> Dict[str, Any]:
    """Check SSL/TLS certificate validity and configuration."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    port = parsed.port or 443
    
    result = {
        "check": "SSL/TLS Certificate",
        "items": []
    }

    if parsed.scheme != "https":
        result["items"].append({
            "severity": WARNING,
            "message": "Site not using HTTPS",
            "detail": f"The site is served over HTTP ({url}). HTTPS is strongly recommended."
        })
        return result

    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                protocol = ssock.version()

                # Check expiration
                not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                days_left = (not_after - datetime.utcnow()).days

                if days_left < 0:
                    result["items"].append({
                        "severity": CRITICAL,
                        "message": "SSL certificate has EXPIRED",
                        "detail": f"Expired {abs(days_left)} days ago on {cert['notAfter']}"
                    })
                elif days_left < 30:
                    result["items"].append({
                        "severity": WARNING,
                        "message": f"SSL certificate expires in {days_left} days",
                        "detail": f"Certificate expires on {cert['notAfter']}"
                    })
                else:
                    result["items"].append({
                        "severity": PASS,
                        "message": f"SSL certificate valid ({days_left} days remaining)",
                        "detail": f"Expires {cert['notAfter']}"
                    })

                # Check protocol version
                if protocol in ('TLSv1', 'TLSv1.1'):
                    result["items"].append({
                        "severity": CRITICAL,
                        "message": f"Outdated TLS protocol: {protocol}",
                        "detail": "TLSv1.0 and TLSv1.1 are deprecated. Use TLSv1.2 or TLSv1.3."
                    })
                else:
                    result["items"].append({
                        "severity": PASS,
                        "message": f"TLS protocol: {protocol}",
                        "detail": "Using a secure TLS version."
                    })

                # Check issuer
                issuer = dict(x[0] for x in cert.get('issuer', []))
                result["items"].append({
                    "severity": INFO,
                    "message": f"Certificate issuer: {issuer.get('organizationName', 'Unknown')}",
                    "detail": f"Issued to: {dict(x[0] for x in cert.get('subject', [])).get('commonName', 'Unknown')}"
                })

    except ssl.SSLCertVerificationError as e:
        result["items"].append({
            "severity": CRITICAL,
            "message": "SSL certificate verification failed",
            "detail": str(e)
        })
    except Exception as e:
        result["items"].append({
            "severity": WARNING,
            "message": "Could not check SSL certificate",
            "detail": str(e)
        })

    return result


async def check_security_headers(url: str) -> Dict[str, Any]:
    """Check for important security headers."""
    result = {
        "check": "Security Headers",
        "items": []
    }

    required_headers = {
        "Strict-Transport-Security": {
            "severity": WARNING,
            "message": "Missing HSTS header",
            "detail": "Strict-Transport-Security header not set. Browsers won't enforce HTTPS."
        },
        "Content-Security-Policy": {
            "severity": WARNING,
            "message": "Missing Content-Security-Policy header",
            "detail": "CSP header not set. The site is more vulnerable to XSS attacks."
        },
        "X-Frame-Options": {
            "severity": WARNING,
            "message": "Missing X-Frame-Options header",
            "detail": "Without X-Frame-Options, the site can be embedded in iframes (clickjacking risk)."
        },
        "X-Content-Type-Options": {
            "severity": INFO,
            "message": "Missing X-Content-Type-Options header",
            "detail": "Should be set to 'nosniff' to prevent MIME-type sniffing."
        },
        "X-XSS-Protection": {
            "severity": INFO,
            "message": "Missing X-XSS-Protection header",
            "detail": "Legacy XSS protection header not set."
        },
        "Referrer-Policy": {
            "severity": INFO,
            "message": "Missing Referrer-Policy header",
            "detail": "Consider setting a Referrer-Policy to control referrer information."
        },
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            response = await client.head(url)
            headers = response.headers

            for header_name, issue in required_headers.items():
                if header_name.lower() in [h.lower() for h in headers.keys()]:
                    result["items"].append({
                        "severity": PASS,
                        "message": f"{header_name} is set",
                        "detail": f"Value: {headers.get(header_name, '')[:100]}"
                    })
                else:
                    result["items"].append(issue)

    except Exception as e:
        result["items"].append({
            "severity": WARNING,
            "message": "Could not check security headers",
            "detail": str(e)
        })

    return result


async def check_broken_links(url: str, html: str, max_links: int = 20) -> Dict[str, Any]:
    """Check for broken links on the page."""
    result = {
        "check": "Broken Links",
        "items": []
    }

    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:") or href.startswith("tel:"):
            continue
        full_url = urljoin(url, href)
        if full_url.startswith("http"):
            links.add(full_url)

    links = list(links)[:max_links]
    broken = 0
    checked = 0

    async def probe(client: httpx.AsyncClient, link: str):
        try:
            return link, (await client.head(link)).status_code
        except Exception:
            return link, None

    # Probe concurrently so large pages don't stall the scan
    async with httpx.AsyncClient(follow_redirects=True, timeout=8) as client:
        probes = await asyncio.gather(*(probe(client, link) for link in links))

    for link, status in probes:
        if status is None:
            broken += 1
            result["items"].append({
                "severity": WARNING,
                "message": "Unreachable link",
                "detail": link[:120]
            })
            continue
        checked += 1
        if status >= 400:
            broken += 1
            result["items"].append({
                "severity": WARNING if status == 404 else INFO,
                "message": f"Broken link (HTTP {status})",
                "detail": link[:120]
            })

    if broken == 0:
        result["items"].append({
            "severity": PASS,
            "message": f"All {checked} links are valid",
            "detail": f"Checked {checked} of {len(links)} discovered links."
        })

    return result


async def check_mixed_content(url: str, html: str) -> Dict[str, Any]:
    """Check for mixed content (HTTP resources on HTTPS page)."""
    result = {
        "check": "Mixed Content",
        "items": []
    }

    parsed = urlparse(url)
    if parsed.scheme != "https":
        result["items"].append({
            "severity": INFO,
            "message": "Site is HTTP — mixed content check not applicable",
            "detail": "Mixed content only applies to HTTPS sites."
        })
        return result

    soup = BeautifulSoup(html, "html.parser")
    mixed = []

    # Check scripts, styles, images, iframes
    for tag in soup.find_all(["script", "link", "img", "iframe", "video", "audio", "source"]):
        src = tag.get("src") or tag.get("href")
        if src and src.startswith("http://"):
            mixed.append({"tag": tag.name, "url": src[:120]})

    if mixed:
        for item in mixed[:10]:
            result["items"].append({
                "severity": WARNING,
                "message": f"Mixed content: <{item['tag']}> loads HTTP resource",
                "detail": item["url"]
            })
    else:
        result["items"].append({
            "severity": PASS,
            "message": "No mixed content detected",
            "detail": "All resources are loaded over HTTPS."
        })

    return result


async def check_exposed_secrets(html: str) -> Dict[str, Any]:
    """Scan page source for exposed secrets and credentials."""
    result = {
        "check": "Exposed Secrets",
        "items": []
    }

    found_secrets = []
    for pattern, secret_type in SECRET_PATTERNS:
        matches = re.findall(pattern, html)
        if matches:
            for match in matches[:3]:
                value = match if isinstance(match, str) else match[-1]
                # Mask the value
                masked = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "****"
                found_secrets.append({
                    "type": secret_type,
                    "masked_value": masked
                })

    if found_secrets:
        for secret in found_secrets[:5]:
            result["items"].append({
                "severity": CRITICAL,
                "message": f"Possible {secret['type']} exposed in source",
                "detail": f"Masked value: {secret['masked_value']}"
            })
    else:
        result["items"].append({
            "severity": PASS,
            "message": "No exposed secrets detected",
            "detail": "No API keys, tokens, or passwords found in page source."
        })

    return result


async def check_libraries(html: str) -> Dict[str, Any]:
    """Detect JavaScript libraries and flag outdated versions."""
    result = {
        "check": "JavaScript Libraries",
        "items": []
    }

    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", src=True)
    script_srcs = [s["src"] for s in scripts]
    all_text = " ".join(script_srcs).lower()

    # Also check inline scripts
    inline = soup.find_all("script", src=False)
    inline_text = " ".join(s.get_text() for s in inline).lower()
    all_text += " " + inline_text

    found_libs = []
    for lib_name, pattern in LIBRARY_PATTERNS.items():
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            version = match.group(1) if match.groups() else "unknown"
            found_libs.append({"library": lib_name, "version": version})

    if found_libs:
        for lib in found_libs:
            # Flag old versions
            severity = INFO
            if lib["library"] == "jquery" and lib["version"].startswith("1."):
                severity = WARNING
            elif lib["library"] == "moment":
                severity = INFO  # moment.js is deprecated

            result["items"].append({
                "severity": severity,
                "message": f"Detected {lib['library']} v{lib['version']}",
                "detail": f"Library: {lib['library']}, Version: {lib['version']}"
            })
    else:
        result["items"].append({
            "severity": INFO,
            "message": "No common JS libraries detected via script tags",
            "detail": "The page may use bundled/minified libraries not detectable by pattern matching."
        })

    return result


# =============================================================================
# Remediation Fixes
# =============================================================================

# (pattern matched against the finding message, short fix, config snippet)
FIX_RULES = [
    (r"not using HTTPS", "Serve the site over HTTPS and redirect all HTTP traffic.",
     "# nginx\nserver {\n  listen 80;\n  server_name example.com;\n  return 301 https://$host$request_uri;\n}\n# Free certificates: certbot --nginx -d example.com"),
    (r"EXPIRED|expires in", "Renew the TLS certificate and enable automatic renewal.",
     "certbot renew --dry-run\n# cron: 0 3 * * * certbot renew --quiet"),
    (r"verification failed", "Install a certificate from a trusted CA that matches the hostname, including the full intermediate chain.",
     "ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;\nssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;"),
    (r"Outdated TLS", "Disable TLS 1.0/1.1 and allow only TLS 1.2+.",
     "ssl_protocols TLSv1.2 TLSv1.3;\nssl_prefer_server_ciphers on;"),
    (r"HSTS", "Add a Strict-Transport-Security header so browsers always use HTTPS.",
     'add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;'),
    (r"Content-Security-Policy", "Define a Content-Security-Policy to restrict where scripts, styles and frames can load from.",
     "add_header Content-Security-Policy \"default-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'self'; base-uri 'self'\" always;"),
    (r"X-Frame-Options", "Prevent clickjacking by forbidding cross-origin framing.",
     'add_header X-Frame-Options "SAMEORIGIN" always;\n# or in CSP: frame-ancestors \'self\''),
    (r"X-Content-Type-Options", "Stop MIME-type sniffing.",
     'add_header X-Content-Type-Options "nosniff" always;'),
    (r"X-XSS-Protection", "Legacy header; rely on CSP, and set it to 0 to disable the buggy legacy auditor.",
     'add_header X-XSS-Protection "0" always;'),
    (r"Referrer-Policy", "Limit how much URL information is leaked to other sites.",
     'add_header Referrer-Policy "strict-origin-when-cross-origin" always;'),
    (r"Broken link|Unreachable link", "Update or remove the dead link, or add a 301 redirect to the new location.",
     "location = /old-path { return 301 /new-path; }"),
    (r"Mixed content", "Load every sub-resource over HTTPS (or protocol-relative) and upgrade insecure requests.",
     "add_header Content-Security-Policy \"upgrade-insecure-requests\" always;"),
    (r"exposed in source", "Revoke and rotate the credential immediately, remove it from client-side code, and keep secrets server-side in environment variables.",
     "# .env (server only, never shipped to the browser)\nAPI_KEY=...\n# scan history: git log -p | grep -i 'api_key'"),
    (r"Detected .* v", "Check this library version against known CVEs and upgrade to the latest patched release.",
     "npm audit\nnpm install <library>@latest"),
    (r"Could not check|Check failed", "Re-run the scan; the host may be blocking automated requests or timing out.", ""),
]


def attach_fixes(checks: List[Dict[str, Any]]) -> None:
    """Attach a `fix` to every non-passing finding, in place."""
    for check in checks:
        for item in check.get("items", []):
            if item.get("severity") == PASS or "fix" in item:
                continue
            for pattern, summary, snippet in FIX_RULES:
                if re.search(pattern, item.get("message", ""), re.I):
                    item["fix"] = {"summary": summary, "snippet": snippet}
                    break


# =============================================================================
# Main Scanner
# =============================================================================

async def scan_url(url: str, check_links: bool = True) -> Dict[str, Any]:
    """
    Run all security checks on a URL.
    
    Returns a comprehensive security report.
    """
    import time
    start = time.time()

    # Fetch the page
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            response = await client.get(url, headers=headers)
            html = response.text
    except Exception as e:
        return {
            "url": url,
            "error": f"Failed to fetch page: {str(e)}",
            "checks": [],
            "summary": {"critical": 0, "warning": 0, "info": 0, "pass": 0}
        }

    # Run all checks concurrently
    tasks = [
        check_ssl(url),
        check_security_headers(url),
        check_mixed_content(url, html),
        check_exposed_secrets(html),
        check_libraries(html),
    ]
    if check_links:
        tasks.append(check_broken_links(url, html))

    checks = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    valid_checks = []
    summary = {"critical": 0, "warning": 0, "info": 0, "pass": 0}

    for check in checks:
        if isinstance(check, Exception):
            valid_checks.append({
                "check": "Error",
                "items": [{"severity": WARNING, "message": "Check failed", "detail": str(check)}]
            })
            summary["warning"] += 1
        else:
            valid_checks.append(check)
            for item in check.get("items", []):
                sev = item.get("severity", INFO)
                if sev in summary:
                    summary[sev] += 1

    attach_fixes(valid_checks)

    # Overall score (0-100)
    score = max(0, 100 - (summary["critical"] * 25) - (summary["warning"] * 10) - (summary["info"] * 2))

    duration = time.time() - start

    return {
        "url": url,
        "score": score,
        "grade": "A" if score >= 90 else "B" if score >= 70 else "C" if score >= 50 else "D" if score >= 30 else "F",
        "checks": valid_checks,
        "summary": summary,
        "duration_seconds": round(duration, 2),
        "scanned_at": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Test
# =============================================================================

async def test_scanner():
    """Test the scanner on a sample URL."""
    result = await scan_url("https://quotes.toscrape.com/")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(test_scanner())
