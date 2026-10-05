"""Generate local TLS certificates and an nginx proxy; no Hailo imports required."""

import argparse
import ipaddress
import json
import os
import plistlib
import re
import secrets
import socket
import subprocess
import uuid
from pathlib import Path


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def names(dns=(), ips=()):
    hostname = socket.gethostname()
    detected = [hostname, hostname.split(".")[0] + ".local", "localhost", *dns]
    addresses = ["127.0.0.1", "::1", *ips]
    # Collect actual interface addresses without resolving the host through DNS.
    try:
        interfaces = json.loads(run("ip", "-j", "address", "show"))
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        interfaces = []
    for interface in interfaces:
        for entry in interface.get("addr_info", []):
            address = entry.get("local", "")
            if entry.get("scope") in {"global", "host"}:
                addresses.append(address)
    checked_dns = []
    for name in detected:
        if len(name) > 253 or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", name):
            raise ValueError(f"Invalid certificate hostname: {name!r}")
        checked_dns.append(name)
    checked_ips = [str(ipaddress.ip_address(address)) for address in addresses]
    return sorted(set(checked_dns)), sorted(set(checked_ips))


def write(path, content, mode=0o644):
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content.encode() if isinstance(content, str) else content)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate(directory, dns=(), ips=(), https_port=8443, http_port=8090, max_body=16777216, timeout=180):
    if not 1 <= https_port <= 65535 or not 1 <= http_port <= 65535 or https_port == http_port:
        raise ValueError("HTTPS and HTTP ports must be distinct and between 1 and 65535")
    if max_body < 1024 or timeout <= 0:
        raise ValueError("Invalid proxy body limit or timeout")
    directory = Path(directory).resolve()
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(directory)):
        raise ValueError("TLS directory must have a simple absolute path")
    dns, ips = names(dns, ips)
    directory.mkdir(parents=True, exist_ok=True, mode=0o755)
    directory.chmod(0o755)
    ca_key, ca_cert = directory / "ca.key", directory / "hailo-ca.crt"
    if ca_key.exists() != ca_cert.exists():
        raise ValueError("Incomplete CA key/certificate pair; restore the missing file before retrying")
    if not ca_key.exists():
        # Only the self-signed CA is installed on clients; its private key never leaves the host.
        try:
            run("openssl", "req", "-x509", "-newkey", "rsa:3072", "-sha256", "-nodes",
                "-days", "3650", "-subj", "/CN=Hailo-10H Local CA", "-keyout", str(ca_key),
                "-out", str(ca_cert), "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign")
        except BaseException:
            ca_key.unlink(missing_ok=True)
            ca_cert.unlink(missing_ok=True)
            raise
    ca_key.chmod(0o600)
    ca_cert.chmod(0o644)
    # Refuse expired CAs or mismatched key pairs instead of silently changing client trust.
    run("openssl", "x509", "-in", str(ca_cert), "-checkend", "2592000", "-noout")
    if run("openssl", "pkey", "-in", str(ca_key), "-pubout") != run("openssl", "x509", "-in", str(ca_cert), "-pubkey", "-noout"):
        raise ValueError("Existing CA certificate does not match its private key")
    key, cert = directory / "server.key", directory / "server.crt"
    meta = {"dns": dns, "ips": ips}
    previous = directory / "names.json"
    reuse = False
    if key.exists() and cert.exists() and previous.exists():
        try:
            reuse = json.loads(previous.read_text()) == meta
            run("openssl", "x509", "-in", str(cert), "-checkend", "2592000", "-noout")
            run("openssl", "verify", "-CAfile", str(ca_cert), str(cert))
            reuse = reuse and run("openssl", "pkey", "-in", str(key), "-pubout") == run("openssl", "x509", "-in", str(cert), "-pubkey", "-noout")
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            reuse = False
    if not reuse:
        temporary_key, temporary_cert = directory / "server.new.key", directory / "server.new.crt"
        csr, extensions = directory / "server.csr", directory / "server.ext"
        san = ",".join([*(f"DNS:{name}" for name in dns), *(f"IP:{ip}" for ip in ips)])
        write(extensions, "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=" + san + "\n")
        try:
            run("openssl", "req", "-new", "-newkey", "rsa:2048", "-nodes", "-sha256",
                "-subj", "/CN=" + dns[0], "-keyout", str(temporary_key), "-out", str(csr))
            temporary_key.chmod(0o600)
            run("openssl", "x509", "-req", "-in", str(csr), "-CA", str(ca_cert), "-CAkey", str(ca_key),
                "-set_serial", "0x" + secrets.token_hex(16), "-days", "365", "-sha256",
                "-extfile", str(extensions), "-out", str(temporary_cert))
            run("openssl", "verify", "-CAfile", str(ca_cert), str(temporary_cert))
            temporary_cert.chmod(0o644)
            temporary_key.replace(key)
            temporary_cert.replace(cert)
            write(previous, json.dumps(meta, indent=2) + "\n")
        finally:
            for temporary in (temporary_key, temporary_cert, csr, extensions):
                temporary.unlink(missing_ok=True)
    key.chmod(0o600)
    der = subprocess.run(["openssl", "x509", "-in", str(ca_cert), "-outform", "DER"], check=True, capture_output=True).stdout
    profile = {
        "PayloadType": "Configuration", "PayloadVersion": 1,
        "PayloadIdentifier": "local.hailo10h.ca.profile", "PayloadUUID": str(uuid.uuid4()),
        "PayloadDisplayName": "Hailo-10H Local CA",
        "PayloadDescription": "Trust the local Hailo HTTPS service. Enable certificate trust after installing.",
        "PayloadContent": [{
            "PayloadType": "com.apple.security.root", "PayloadVersion": 1,
            "PayloadIdentifier": "local.hailo10h.ca.certificate", "PayloadUUID": str(uuid.uuid4()),
            "PayloadDisplayName": "Hailo-10H Local CA", "PayloadContent": der,
        }],
    }
    write(directory / "hailo-ca.mobileconfig", plistlib.dumps(profile))
    # Dedicated server block: HTTP API, SSE, WebSocket and MCP use the existing runtime.
    configuration = f"""server {{
    listen {https_port} ssl;
    server_name _;
    ssl_certificate {cert};
    ssl_certificate_key {key};
    ssl_protocols TLSv1.2 TLSv1.3;
    client_max_body_size {max_body};
    location = /hailo-ca.crt {{
        alias {ca_cert};
        default_type application/x-x509-ca-cert;
    }}
    location = /hailo-ca.mobileconfig {{
        alias {directory / 'hailo-ca.mobileconfig'};
        default_type application/x-apple-aspen-config;
    }}
    location / {{
        proxy_pass http://127.0.0.1:{http_port};
        proxy_http_version 1.1;
        proxy_set_header Host $http_host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $hailo_connection_upgrade;
        proxy_buffering off;
        proxy_read_timeout {max(300, int(timeout) + 30)}s;
        proxy_send_timeout {max(300, int(timeout) + 30)}s;
    }}
}}
"""
    write(directory / "nginx-site.conf", "map $http_upgrade $hailo_connection_upgrade {\n    default upgrade;\n    '' close;\n}\n" + configuration)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="/etc/hailo-10h-services/tls")
    parser.add_argument("--dns", action="append", default=[])
    parser.add_argument("--ip", action="append", default=[])
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--http-port", type=int, default=8090)
    parser.add_argument("--env-file", default="/etc/hailo-10h-services.env")
    parser.add_argument("--config", default="/etc/hailo-10h-services.yaml")
    args = parser.parse_args()
    # Read only numeric proxy settings. Do not source/execute the service's env file.
    values = {}
    if Path(args.config).exists():
        import yaml
        document = yaml.safe_load(Path(args.config).read_text()) or {}
        for key in ("port", "max_body", "request_timeout", "host"):
            if key in document.get("settings", {}):
                values["HAILO_" + key.upper()] = str(document["settings"][key])
    if Path(args.env_file).exists():
        for line in Path(args.env_file).read_text().splitlines():
            key, _, value = line.partition("=")
            if key in {"HAILO_PORT", "HAILO_MAX_BODY", "HAILO_REQUEST_TIMEOUT", "HAILO_HOST"}:
                values[key] = value.strip().strip("\"'")
    if values.get("HAILO_HOST", "0.0.0.0") not in {"0.0.0.0", "127.0.0.1"}:
        parser.error("HTTPS proxy needs HAILO_HOST=0.0.0.0 or 127.0.0.1")
    old_umask = os.umask(0o077)
    try:
        meta = generate(args.directory, args.dns, args.ip, args.port,
                        int(values.get("HAILO_PORT", args.http_port)),
                        int(values.get("HAILO_MAX_BODY", 16777216)),
                        float(values.get("HAILO_REQUEST_TIMEOUT", 180)))
    finally:
        os.umask(old_umask)
    print("Certificate DNS names: " + ", ".join(meta["dns"]))
    print("Certificate IP addresses: " + ", ".join(meta["ips"]))
    print(run("openssl", "x509", "-in", str(Path(args.directory) / "hailo-ca.crt"), "-noout", "-fingerprint", "-sha256").strip())


if __name__ == "__main__":
    main()
