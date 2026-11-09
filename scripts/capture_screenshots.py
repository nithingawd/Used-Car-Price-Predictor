"""Capture reports/figures/gui_*.png of every page at desktop and narrow widths, light and dark.

Drives a headless Chrome over the DevTools protocol so it can wait until Streamlit has
finished rendering (no skeletons, no running spinners) before capturing.

Usage (with the app already running):
    streamlit run app/Home.py --server.headless true --server.port 8765
    python scripts/capture_screenshots.py --base-url http://localhost:8765
"""

import argparse
import asyncio
import base64
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websockets

# Standard install locations, tried after the CHROME_PATH environment variable and PATH lookup.
CHROME_CANDIDATES = [
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
]
PAGES = {
    "estimator": "",
    "market_explorer": "market_explorer",
    "model_performance": "model_performance",
    "about_and_data": "about_and_data",
}
WIDTHS = {"desktop": 1440, "narrow": 800}
# The app follows the browser colour scheme by default (D-036); dark shots get a "_dark" suffix.
SCHEMES = {"light": "", "dark": "_dark"}
DEBUG_PORT = 9333
BROWSER_NAMES = ("chrome", "google-chrome", "chromium", "msedge")

logger = logging.getLogger("capture_screenshots")
READY_SCRIPT = """
(() => {
  const app = document.querySelector('[data-testid="stApp"]');
  const busy = document.querySelector('[data-testid="stSkeleton"], [data-testid="stSpinner"]');
  const running = document.querySelector('[data-testid="stStatusWidget"]');
  const h1 = document.querySelector('h1.cp-title');
  const charts = [...document.querySelectorAll('[data-testid="stPlotlyChart"]')];
  const drawn = charts.every(c => c.querySelector('.main-svg'));
  const images = [...document.querySelectorAll('img')].every(i => i.complete);
  return Boolean(app && h1 && !busy && !running && drawn && images)
    ? document.querySelectorAll('[data-testid="stElementContainer"]').length : 0;
})()
"""
HEIGHT_SCRIPT = """
(() => {
  const main = document.querySelector('[data-testid="stMain"]') || document.body;
  return Math.max(main.scrollHeight, document.body.scrollHeight);
})()
"""


class DevTools:
    """Minimal Chrome DevTools protocol client over one page websocket."""

    def __init__(self, socket: websockets.ClientConnection) -> None:
        self.socket, self.counter = socket, 0

    async def call(self, method: str, **params: object) -> dict:
        """Send one command and wait for its result."""
        self.counter += 1
        await self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params}))
        while True:
            message = json.loads(await self.socket.recv())
            if message.get("id") == self.counter:
                return message.get("result", {})

    async def evaluate(self, expression: str) -> object:
        """Evaluate JavaScript and return its value."""
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return result.get("result", {}).get("value")


async def wait_until_ready(tools: DevTools, timeout_s: float) -> None:
    """Poll until the page has rendered, then let animations settle."""
    deadline = time.monotonic() + timeout_s
    previous = 0
    while time.monotonic() < deadline:
        count = await tools.evaluate(READY_SCRIPT)
        if count and count == previous:
            await asyncio.sleep(1.5)
            return
        previous = count or 0
        await asyncio.sleep(2.0)
    raise TimeoutError("Page did not finish rendering in time")


async def capture(tools: DevTools, url: str, width: int, scheme: str, out_path: Path) -> None:
    """Load url at the given width and colour scheme and save a full-height PNG."""
    await tools.call(
        "Emulation.setDeviceMetricsOverride",
        width=width,
        height=1000,
        deviceScaleFactor=1,
        mobile=False,
    )
    await tools.call(
        "Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": scheme}]
    )
    await tools.call("Page.navigate", url=url)
    await wait_until_ready(tools, timeout_s=90)
    height = int(await tools.evaluate(HEIGHT_SCRIPT) or 1000) + 120
    await tools.call(
        "Emulation.setDeviceMetricsOverride",
        width=width,
        height=height,
        deviceScaleFactor=1,
        mobile=False,
    )
    await asyncio.sleep(1.5)
    shot = await tools.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
    out_path.write_bytes(base64.b64decode(shot["data"]))


def find_browser() -> Path:
    """Locate Chrome or Edge: $CHROME_PATH, then PATH, then the standard install folders.

    Raises:
        FileNotFoundError: if no Chromium-based browser is found.
    """
    override = os.environ.get("CHROME_PATH")
    found = [Path(override)] if override else []
    found += [Path(p) for name in BROWSER_NAMES if (p := shutil.which(name))]
    browser = next((p for p in [*found, *CHROME_CANDIDATES] if p.exists()), None)
    if browser is None:
        raise FileNotFoundError("No Chrome or Edge found; set CHROME_PATH to its executable")
    return browser


def start_browser(profile: str) -> subprocess.Popen:
    """Launch headless Chrome (or Edge) with remote debugging."""
    browser = find_browser()
    return subprocess.Popen(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def page_socket_url() -> str:
    """WebSocket URL of the first page target (waits for the browser to come up)."""
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json") as response:
                targets = json.load(response)
            return next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        except (OSError, StopIteration):
            time.sleep(0.5)
    raise TimeoutError("Browser did not expose a page target")


async def run(base_url: str, out_dir: Path) -> list[Path]:
    """Capture every page at every width."""
    saved = []
    async with websockets.connect(page_socket_url(), max_size=None) as socket:
        tools = DevTools(socket)
        await tools.call("Page.enable")
        for scheme, suffix in SCHEMES.items():
            for name, path in PAGES.items():
                for label, width in WIDTHS.items():
                    target = out_dir / f"gui_{name}_{label}{suffix}.png"
                    url = f"{base_url.rstrip('/')}/{path}"
                    await capture(tools, url, width, scheme, target)
                    saved.append(target)
                    logger.info("saved %s", target)
        await tools.call("Page.navigate", url="about:blank")  # close the app session cleanly
        await asyncio.sleep(1.0)
    return saved


def main() -> None:
    """Parse arguments, start the browser, capture, and always close the browser."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8765")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        browser = start_browser(profile)
        try:
            asyncio.run(run(args.base_url, args.out_dir))
        finally:
            browser.terminate()
            browser.wait(timeout=10)


if __name__ == "__main__":
    main()
