import os
import shlex
import subprocess
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:
    # dotenv is optional; provide a noop if not installed so the package still imports
    def load_dotenv(*args, **kwargs):
        return False


def _from_env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _from_dotenv(name: str) -> Optional[str]:
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        return os.environ.get(name)
    return None


def _from_keychain(name: str) -> Optional[str]:
    # macOS 'security' CLI
    try:
        cmd = ["security", "find-generic-password", "-s", name, "-w"]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def get_secret(name: str) -> Optional[str]:
    """Retrieve a secret by checking: ENV -> .env -> macOS Keychain."""
    val = _from_env(name)
    if val:
        return val
    val = _from_dotenv(name)
    if val:
        return val
    val = _from_keychain(name)
    return val


def set_env_from_secret(name: str) -> Optional[str]:
    """Set environment variable from secret and return the value (or None).

    Returns the secret string when found and set, otherwise None.
    """
    val = get_secret(name)
    if val:
        os.environ[name] = val
        return val
    return None
