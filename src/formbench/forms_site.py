"""LocalForms (FormFactory-style Flask site) lifecycle."""

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from formbench.common import FormbenchError, info
from formbench.settings import Settings

APP_PATH = Path("evaluation_additions/formfactory_import/site/app.py")


def site_reachable(settings: Settings, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(settings.localforms_base_url + "/", timeout=timeout) as response:
            return response.status < 500
    except (urllib.error.URLError, OSError):
        return False


def serve_command(settings: Settings) -> list:
    return [
        str(settings.python_bin), str(settings.root / APP_PATH),
        "--host", settings.localforms_host, "--port", str(settings.localforms_port),
    ]


class LocalFormsSite:
    """Reuse a running site or start one for the duration of a `with` block."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: Optional[subprocess.Popen] = None
        self.log_path = settings.logs_dir / "serve" / f"localforms-{settings.localforms_port}.log"

    def __enter__(self) -> "LocalFormsSite":
        if site_reachable(self.settings):
            info(f"reusing LocalForms site at {self.settings.localforms_base_url}")
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        info(f"starting LocalForms site at {self.settings.localforms_base_url} (log: {self.log_path})")
        with open(self.log_path, "ab") as log:
            self.process = subprocess.Popen(
                serve_command(self.settings), stdout=log, stderr=subprocess.STDOUT,
                env=self.settings.subprocess_env(), cwd=str(self.settings.root), start_new_session=True,
            )
        for _ in range(30):
            if site_reachable(self.settings):
                return self
            if self.process.poll() is not None:
                break
            time.sleep(2)
        self.__exit__()
        raise FormbenchError(
            f"LocalForms site did not start on {self.settings.localforms_base_url}",
            hint=f"see {self.log_path}; is flask installed (make doctor) and the port free (LOCALFORMS_PORT)?",
        )

    def __exit__(self, *exc: object) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
