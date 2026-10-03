import plistlib
import ssl
import subprocess

import pytest

from hailo_services.tls import generate, names


def test_certificates_names_profile_and_key_permissions(tmp_path):
    directory = tmp_path / 'tls'
    meta = generate(directory, dns=['cm5.local'], ips=['192.168.1.42'])
    assert 'cm5.local' in meta['dns'] and '192.168.1.42' in meta['ips']
    subprocess.run(['openssl', 'verify', '-CAfile', str(directory / 'hailo-ca.crt'),
                    '-verify_ip', '192.168.1.42', str(directory / 'server.crt')], check=True)
    subprocess.run(['openssl', 'verify', '-CAfile', str(directory / 'hailo-ca.crt'),
                    '-verify_hostname', 'cm5.local', str(directory / 'server.crt')], check=True)
    assert (directory / 'ca.key').stat().st_mode & 0o777 == 0o600
    assert (directory / 'server.key').stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o755
    profile = plistlib.loads((directory / 'hailo-ca.mobileconfig').read_bytes())
    der = ssl.PEM_cert_to_DER_cert((directory / 'hailo-ca.crt').read_text())
    assert profile['PayloadContent'][0]['PayloadContent'] == der
    assert profile['PayloadContent'][0]['PayloadType'] == 'com.apple.security.root'
    before = {name: (directory / name).read_bytes() for name in ['ca.key', 'hailo-ca.crt', 'server.crt']}
    generate(directory, dns=['cm5.local'], ips=['192.168.1.42'])
    assert before == {name: (directory / name).read_bytes() for name in before}
    generate(directory, dns=['cm5.local'], ips=['192.168.1.43'])
    assert (directory / 'hailo-ca.crt').read_bytes() == before['hailo-ca.crt']
    assert (directory / 'server.crt').read_bytes() != before['server.crt']
    config = (directory / 'nginx-site.conf').read_text()
    assert 'listen 8443 ssl;' in config and 'proxy_pass http://127.0.0.1:8090;' in config
    assert 'proxy_buffering off;' in config and 'proxy_set_header Upgrade $http_upgrade;' in config
    assert 'location = /hailo-ca.mobileconfig' in config
    assert 'ca.key' not in config


def test_tls_input_validation_and_incomplete_ca(tmp_path):
    with pytest.raises(ValueError):
        names(dns=['bad; ssl_certificate injected;'])
    with pytest.raises(ValueError):
        names(ips=['192.168.1.999'])
    with pytest.raises(ValueError):
        generate(tmp_path, https_port=8090)
    (tmp_path / 'ca.key').write_text('incomplete')
    with pytest.raises(ValueError, match='Incomplete CA'):
        generate(tmp_path)


def test_real_https_listener_verifies_certificate_and_hostname(tmp_path):
    import http.server
    import threading
    import urllib.request

    generate(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / 'server.crt', tmp_path / 'server.key')

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'HTTPS OK')

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        client_context = ssl.create_default_context(cafile=str(tmp_path / 'hailo-ca.crt'))
        with urllib.request.urlopen(f'https://localhost:{server.server_port}/', context=client_context) as response:
            assert response.read() == b'HTTPS OK'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
