"""Validated, atomic YAML edits for the authenticated settings UI."""

import hashlib
import os
import stat
import tempfile
import threading
from pathlib import Path

import yaml

from hailo_services.config import Settings

_LOCK = threading.RLock()
_SECRET = {"settings.api_key", "settings.mqtt_password"}


class SettingsStoreError(Exception):
    """Represent a safe HTTP error for configuration operations."""

    def __init__(self, message, status=400):
        """Create a failure with its HTTP status.

        Returns:
            None: Initializes the exception.
        """
        super().__init__(message)
        self.status = status


class SettingsStore:
    """Read and update the service configuration atomically."""

    def __init__(self, path=None):
        """Choose the same YAML path as the running service.

        Returns:
            None: Initializes the store.
        """
        self.path = Path(path or os.getenv("HAILO_CONFIG") or "/etc/hailo-10h-services.yaml")

    def _load(self):
        """Read YAML and its exact bytes.

        Returns:
            tuple: Raw bytes and mapping.
        """
        try:
            raw = self.path.read_bytes()
            document = yaml.safe_load(raw)
        except (OSError, yaml.YAMLError) as exc:
            raise SettingsStoreError(f"Konfiguration kann nicht gelesen werden: {exc}", 503) from exc
        if not isinstance(document, dict):
            raise SettingsStoreError("Die YAML-Wurzel muss eine Zuordnung sein.")
        return raw, document

    @staticmethod
    def _fields(document):
        """Flatten editable scalar leaves while hiding secrets.

        Returns:
            list: Fields for the browser.
        """
        fields = []
        def visit(node, prefix=""):
            """Traverse nested YAML maps.

            Returns:
                None: Appends scalar fields.
            """
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    visit(value, path)
                elif value is None or type(value) in (str, int, float, bool):
                    fields.append({"path": path, "type": type(value).__name__,
                                   "value": None if path in _SECRET else value,
                                   "secret_set": bool(value) if path in _SECRET else False})
        visit(document)
        return fields

    def read(self):
        """Return a versioned, redacted configuration.

        Returns:
            dict: Public editing metadata for authenticated clients.
        """
        with _LOCK:
            raw, document = self._load()
            return {"version": hashlib.sha256(raw).hexdigest(), "fields": self._fields(document),
                    "path": str(self.path), "restart_available": bool(os.getenv("INVOCATION_ID")),
                    "restart_required": True}

    def write(self, payload):
        """Validate a sparse patch and replace the YAML file.

        Returns:
            dict: New version and editing metadata.
        """
        if not isinstance(payload, dict) or set(payload) != {"version", "changes"}:
            raise SettingsStoreError("Version und Änderungen sind erforderlich.")
        changes = payload["changes"]
        if not isinstance(changes, dict) or len(changes) > 150:
            raise SettingsStoreError("Ungültige Änderungen.")
        with _LOCK:
            raw, document = self._load()
            if payload["version"] != hashlib.sha256(raw).hexdigest():
                raise SettingsStoreError("Die Datei wurde inzwischen geändert. Bitte neu laden.", 409)
            known = {field["path"]: field for field in self._fields(document)}
            if set(changes) - set(known):
                raise SettingsStoreError("Unbekannter Einstellungspfad.")
            for path, value in changes.items():
                field = known[path]
                original = field["type"]
                if path in _SECRET and value == "":
                    continue  # Empty password fields keep the existing secret.
                if path in _SECRET and value is None:
                    value = ""  # Explicit clear from the settings UI.
                if original == "NoneType" and value == "":
                    value = None
                elif original == "float" and type(value) in (int, float):
                    value = float(value)
                elif type(value).__name__ != original:
                    raise SettingsStoreError(f"Ungültiger Typ für {path}: erwartet {original}.")
                cursor = document
                *parents, leaf = path.split(".")
                for name in parents:
                    cursor = cursor[name]
                cursor[leaf] = value
            encoded = yaml.safe_dump(document, allow_unicode=True, sort_keys=False).encode()
            # Validate with exactly the same loader as the next service start,
            # without environment overrides masking broken YAML values.
            candidate = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".hailo-config-", delete=False) as stream:
                    candidate = Path(stream.name)
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                checked = Settings.from_env(path=candidate, use_env=False)
                if checked.vlm_enabled and checked.hailo_llm_enabled:
                    raise ValueError("VLM und Hailo LLM können nicht gleichzeitig aktiv sein")
                if not changes:
                    return self.read()
                current = self.path.stat()
                os.chmod(candidate, stat.S_IMODE(current.st_mode))
                os.chown(candidate, current.st_uid, current.st_gid)
                # The file may have been edited externally while validation ran.
                if hashlib.sha256(self.path.read_bytes()).hexdigest() != payload["version"]:
                    raise SettingsStoreError("Die Datei wurde inzwischen geändert. Bitte neu laden.", 409)
                os.replace(candidate, self.path)
                candidate = None
                fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                if isinstance(exc, SettingsStoreError):
                    raise
                raise SettingsStoreError(f"Speichern/Validierung fehlgeschlagen: {exc}", 400) from exc
            finally:
                if candidate is not None:
                    candidate.unlink(missing_ok=True)
            return self.read()
