"""Request checks for serving Tangent beyond localhost.

The app has no login, so these checks are about keeping *other web pages* in
your browser (and DNS-rebinding tricks) from driving the server, not about
users on the network:

* Host header: only IP addresses, "localhost" and this machine's own names are
  accepted. A DNS-rebinding page arrives with its own domain in the Host
  header and is refused. Extra names can be allowed with --allow-host or the
  TANGENT_ALLOWED_HOSTS environment variable (comma separated).
* Cross-site requests: anything that changes state (POST and friends) must
  come from the app's own page. A request whose Origin names another site, or
  that the browser marks Sec-Fetch-Site: cross-site, is refused. Tools that
  send neither header (curl, scripts) are allowed.
* JSON endpoints only accept a real application/json body (see app.py), which
  browsers cannot send cross-site without a CORS preflight the server never
  grants.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

from flask import jsonify, request

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _split_host(value: str) -> str:
    """Hostname from a Host header or URL netloc (drops the port, brackets)."""
    value = (value or "").strip().lower()
    if value.startswith("["):                       # [::1]:5000
        return value[1:value.find("]")] if "]" in value else value
    if value.count(":") == 1:                       # name:port or v4:port
        return value.split(":")[0]
    return value


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def machine_names():
    names = {"localhost"}
    try:
        hn = socket.gethostname().lower()
        names |= {hn, hn + ".local"}
        names.add(socket.getfqdn().lower())
    except OSError:
        pass
    return names


class RequestGuard:
    def __init__(self, extra_hosts=()):
        env = os.environ.get("TANGENT_ALLOWED_HOSTS", "")
        self.allowed = machine_names() | {h.strip().lower() for h in env.split(",") if h.strip()}
        self.allowed |= {h.strip().lower() for h in extra_hosts if h.strip()}

    def host_ok(self, host: str) -> bool:
        h = _split_host(host)
        return bool(h) and (_is_ip(h) or h in self.allowed)

    def check(self):
        """Flask before_request hook: returns an error response or None."""
        if not self.host_ok(request.host):
            return jsonify({"error": "Host not allowed. Open the app by IP address or this "
                                     "machine's name, or start it with --allow-host <name>."}), 403
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if request.headers.get("Sec-Fetch-Site", "") == "cross-site":
                return jsonify({"error": "Cross-site request refused."}), 403
            origin = request.headers.get("Origin")
            if origin and origin != "null":
                if _split_host(urlsplit(origin).netloc) != _split_host(request.host) or \
                        (urlsplit(origin).port or None) != (_port(request.host)):
                    return jsonify({"error": "Cross-origin request refused."}), 403
            elif origin == "null":
                return jsonify({"error": "Request from an opaque origin refused."}), 403
        return None


def _port(host: str):
    host = host or ""
    if host.startswith("["):
        rest = host[host.find("]") + 1:]
        return int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else None
    if host.count(":") == 1:
        p = host.split(":")[1]
        return int(p) if p.isdigit() else None
    return None


def is_local_bind(host: str) -> bool:
    return (host or "").strip().lower() in LOCAL_HOSTS
