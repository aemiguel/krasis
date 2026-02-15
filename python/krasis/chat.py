"""Krasis Chat — interactive terminal chat client for Krasis servers.

Discovers running Krasis servers on localhost, shows a selection screen,
then provides an interactive streaming chat interface.

Usage:
    python -m krasis.chat                         # auto-discover local servers
    python -m krasis.chat --port 8080             # connect to specific port
    python -m krasis.chat --url http://host:port  # connect to URL
"""

import argparse
import http.client
import json
import os
import select
import signal
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Enable readline for input() — gives arrow keys, history, Ctrl-A/E, etc.
try:
    import readline  # noqa: F401
except ImportError:
    pass

# Terminal handling
try:
    import termios
    import tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False


# ═══════════════════════════════════════════════════════════════════════
# ANSI helpers
# ═══════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"

KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_ENTER = "ENTER"
KEY_ESCAPE = "ESC"
KEY_QUIT = "q"

# Bracketed paste escape sequences
_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"


def _read_input_with_paste() -> str:
    """Read user input supporting multi-line paste via bracketed paste mode.

    - Enter submits (when not in a paste)
    - Pasted text with newlines is captured as-is (terminal sends paste brackets)
    - Ctrl-C raises KeyboardInterrupt, Ctrl-D raises EOFError
    - Basic line editing: backspace works, but no readline features (arrow keys, etc.)
    """
    if not _HAS_TERMIOS:
        return input()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buf = []
    in_paste = False
    esc_buf = ""  # buffer for escape sequence detection

    # Enable bracketed paste mode
    sys.stdout.write("\x1b[?2004h")
    sys.stdout.flush()

    try:
        tty.setcbreak(fd)  # cbreak: chars available immediately, Ctrl-C still works

        while True:
            ch = sys.stdin.read(1)
            if not ch:
                raise EOFError

            # ── Escape sequence detection ──
            if esc_buf or ch == "\x1b":
                esc_buf += ch
                # Check if we have a complete paste bracket
                if esc_buf == _PASTE_START:
                    in_paste = True
                    esc_buf = ""
                    continue
                if esc_buf == _PASTE_END:
                    in_paste = False
                    esc_buf = ""
                    continue
                # Still building — check if it could still become a paste bracket
                if _PASTE_START.startswith(esc_buf) or _PASTE_END.startswith(esc_buf):
                    continue
                # Not a paste bracket — flush esc_buf as literal chars
                # (skip escape sequences we don't handle, like arrow keys)
                esc_buf = ""
                continue

            # ── Ctrl-C ──
            if ch == "\x03":
                sys.stdout.write("\n")
                sys.stdout.flush()
                raise KeyboardInterrupt

            # ── Ctrl-D (EOF) ──
            if ch == "\x04":
                if not buf:
                    raise EOFError
                continue  # ignore if buffer non-empty

            # ── Backspace ──
            if ch in ("\x7f", "\x08"):
                if buf:
                    buf.pop()
                    # Erase character on screen
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            # ── Enter / newline ──
            if ch in ("\r", "\n"):
                if in_paste:
                    # Inside a paste: keep the newline
                    buf.append("\n")
                    sys.stdout.write("\n       ")  # visual continuation indent
                    sys.stdout.flush()
                    continue
                else:
                    # Normal Enter: submit
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(buf)

            # ── Normal character ──
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()

    finally:
        # Disable bracketed paste mode and restore terminal
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def _hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def _show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def _read_key() -> str:
    """Read a single keypress in raw mode."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return KEY_UP
                elif ch3 == "B":
                    return KEY_DOWN
            return KEY_ESCAPE
        elif ch in ("\r", "\n"):
            return KEY_ENTER
        elif ch == "\x03":
            return KEY_ESCAPE
        else:
            return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ═══════════════════════════════════════════════════════════════════════
# Server discovery
# ═══════════════════════════════════════════════════════════════════════

def discover_servers(
    host: str = "localhost",
    ports: Optional[range] = None,
) -> List[Dict[str, Any]]:
    """Scan ports for running Krasis servers."""
    if ports is None:
        ports = range(8080, 8091)

    servers = []
    for port in ports:
        base = f"http://{host}:{port}"
        try:
            req = urllib.request.Request(f"{base}/health")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                health = json.loads(resp.read())
            status = health.get("status", "unknown")

            # Get model name
            model_name = "unknown"
            try:
                req = urllib.request.Request(f"{base}/v1/models")
                with urllib.request.urlopen(req, timeout=0.5) as resp:
                    data = json.loads(resp.read())
                    models_data = data.get("data", [])
                    if models_data:
                        model_name = models_data[0].get("id", "unknown")
            except Exception:
                pass

            servers.append({
                "host": host,
                "port": port,
                "url": base,
                "model": model_name,
                "status": status,
            })
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, ConnectionRefusedError):
            continue

    return servers


# ═══════════════════════════════════════════════════════════════════════
# Server selection screen
# ═══════════════════════════════════════════════════════════════════════

def _server_selection_screen(servers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Arrow-key server picker. Returns selected server or None."""
    if not servers:
        return None

    cursor = 0
    while True:
        _clear_screen()
        lines = []
        lines.append(f"  {BOLD}Select a Krasis server:{NC}\n")

        for i, s in enumerate(servers):
            prefix = f"  {CYAN}\u25b8{NC} " if i == cursor else "    "
            hl = BOLD if i == cursor else ""

            if s["status"] == "ok":
                status_tag = f"{GREEN}ready{NC}"
            elif s["status"] == "loading":
                status_tag = f"{YELLOW}loading...{NC}"
            else:
                status_tag = f"{RED}{s['status']}{NC}"

            lines.append(
                f"{prefix}{hl}{s['host']}:{s['port']}{NC}"
                f"  {s['model']}  ({status_tag})"
            )

        lines.append(f"\n  {DIM}[\u2191\u2193] Select  [Enter] Connect  [q] Quit{NC}")

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        key = _read_key()
        if key == KEY_UP:
            cursor = (cursor - 1) % len(servers)
        elif key == KEY_DOWN:
            cursor = (cursor + 1) % len(servers)
        elif key == KEY_ENTER:
            return servers[cursor]
        elif key == KEY_QUIT or key == KEY_ESCAPE:
            return None


# ═══════════════════════════════════════════════════════════════════════
# Streaming chat via http.client (no extra deps)
# ═══════════════════════════════════════════════════════════════════════

def _parse_host_port(url: str):
    """Parse URL into (host, port, ssl)."""
    if url.startswith("https://"):
        rest, ssl = url[8:], True
    elif url.startswith("http://"):
        rest, ssl = url[7:], False
    else:
        rest, ssl = url, False

    # Strip path
    rest = rest.split("/")[0]

    if ":" in rest:
        host, port_str = rest.rsplit(":", 1)
        return host, int(port_str), ssl
    return rest, (443 if ssl else 80), ssl


def stream_chat(
    url: str,
    messages: List[Dict],
    temperature: float = 0.6,
    max_tokens: int = 4096,
    top_p: float = 0.95,
) -> str:
    """Send streaming chat request. Prints tokens as they arrive. Returns full text."""
    host, port, ssl = _parse_host_port(url)

    body = json.dumps({
        "model": "krasis",
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }).encode("utf-8")

    if ssl:
        conn = http.client.HTTPSConnection(host, port, timeout=600)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=600)

    try:
        conn.request("POST", "/v1/chat/completions", body, {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })
        resp = conn.getresponse()

        if resp.status != 200:
            error_body = resp.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {error_body[:500]}")

        full_text = ""
        # Set a short socket timeout so readline() yields to KeyboardInterrupt
        sock = resp.fp.raw._sock if hasattr(resp.fp, 'raw') else None
        if sock is not None:
            sock.settimeout(0.5)

        while True:
            try:
                raw_line = resp.readline()
            except (TimeoutError, OSError):
                # Socket timeout — check for interrupt and retry
                continue
            if not raw_line:
                break

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if not line.startswith("data: "):
                continue

            payload = line[6:]
            if payload == "[DONE]":
                break

            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                full_text += content

            if choices[0].get("finish_reason"):
                break

        return full_text
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Chat loop
# ═══════════════════════════════════════════════════════════════════════

def chat_loop(
    server: Dict[str, Any],
    temperature: float = 0.6,
    max_tokens: int = 4096,
    system_prompt: str = "",
):
    """Interactive chat loop with streaming responses."""
    url = server["url"]
    model = server["model"]
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    _clear_screen()

    # Banner
    model_display = model[:40]
    print(f"  {BOLD}\u2554{'═' * 50}\u2557{NC}")
    print(f"  {BOLD}\u2551  {CYAN}Krasis Chat{NC}{BOLD} \u2014 {model_display:<37s}\u2551{NC}")
    print(f"  {BOLD}\u255a{'═' * 50}\u255d{NC}")
    print(f"  {DIM}Server: {server['host']}:{server['port']}{NC}")
    print(f"  {DIM}Commands: /new (clear history)  /system <msg>  /exit{NC}")
    print()

    while True:
        try:
            sys.stdout.write(f"  {GREEN}{BOLD}You:{NC} ")
            sys.stdout.flush()
            user_input = _read_input_with_paste()

            if not user_input.strip():
                continue

            stripped = user_input.strip()

            # ── Commands ──
            if stripped.lower() in ("/exit", "/quit", "exit", "quit"):
                print(f"\n  {DIM}Goodbye!{NC}")
                break

            if stripped.lower() == "/new":
                messages.clear()
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                print(f"  {DIM}Conversation cleared.{NC}\n")
                continue

            if stripped.lower().startswith("/system "):
                system_prompt = stripped[8:].strip()
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] = system_prompt
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt})
                print(f"  {DIM}System prompt set.{NC}\n")
                continue

            # ── Send message ──
            messages.append({"role": "user", "content": user_input})

            sys.stdout.write(f"\n  {CYAN}{BOLD}Assistant:{NC} ")
            sys.stdout.flush()

            try:
                t0 = time.perf_counter()
                response_text = stream_chat(
                    url, messages, temperature, max_tokens,
                )
                elapsed = time.perf_counter() - t0

                messages.append({"role": "assistant", "content": response_text})

                # Rough token estimate for stats
                approx_tokens = max(1, len(response_text) // 4)
                if elapsed > 0:
                    tps = approx_tokens / elapsed
                    print(
                        f"\n  {DIM}(~{approx_tokens} tokens, "
                        f"{elapsed:.1f}s, ~{tps:.1f} tok/s){NC}\n"
                    )
                else:
                    print("\n")

            except (ConnectionRefusedError, OSError) as e:
                print(f"\n\n  {RED}Connection lost: {e}{NC}")
                print(f"  {DIM}Server may have shut down.{NC}\n")
                messages.pop()  # remove failed user message
            except RuntimeError as e:
                print(f"\n\n  {RED}Error: {e}{NC}\n")
                messages.pop()
            except Exception as e:
                print(f"\n\n  {RED}{type(e).__name__}: {e}{NC}\n")
                messages.pop()

        except KeyboardInterrupt:
            print(f"\n\n  {DIM}Goodbye!{NC}")
            break
        except EOFError:
            print(f"\n  {DIM}Goodbye!{NC}")
            break


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Ensure Ctrl-C always terminates cleanly (even during blocking I/O)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    parser = argparse.ArgumentParser(
        description="Krasis Chat \u2014 interactive streaming chat client",
    )
    parser.add_argument("--url", default=None,
                        help="Server URL (e.g. http://localhost:8080)")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port on localhost")
    parser.add_argument("--host", default="localhost",
                        help="Server hostname (default: localhost)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--system", default="",
                        help="Initial system prompt")
    parser.add_argument("--scan-start", type=int, default=8080,
                        help="Port scan range start (default: 8080)")
    parser.add_argument("--scan-end", type=int, default=8090,
                        help="Port scan range end (default: 8090)")
    args = parser.parse_args()

    # ── Direct URL connection ──
    if args.url:
        url = args.url.rstrip("/")
        server = {
            "host": "", "port": 0, "url": url,
            "model": "unknown", "status": "ok",
        }
        # Try to fetch model name
        try:
            req = urllib.request.Request(f"{url}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models_data = data.get("data", [])
                if models_data:
                    server["model"] = models_data[0].get("id", "unknown")
            hp = url.split("//", 1)[-1].split("/")[0]
            if ":" in hp:
                server["host"] = hp.rsplit(":", 1)[0]
                server["port"] = int(hp.rsplit(":", 1)[1])
            else:
                server["host"] = hp
                server["port"] = 80
        except Exception:
            pass
        chat_loop(server, args.temperature, args.max_tokens, args.system)
        return

    # ── Port scan ──
    if args.port:
        ports = range(args.port, args.port + 1)
    else:
        ports = range(args.scan_start, args.scan_end + 1)

    print(f"  {DIM}Scanning for Krasis servers on {args.host}:{args.scan_start}-{args.scan_end}...{NC}")
    servers = discover_servers(args.host, ports)

    if not servers:
        print(f"\n  {RED}No Krasis servers found.{NC}")
        print(f"  {DIM}Start one with: ./krasis{NC}")
        sys.exit(1)

    if len(servers) == 1:
        # Single server — connect directly
        server = servers[0]
        status = (
            f"{GREEN}ready{NC}" if server["status"] == "ok"
            else f"{YELLOW}{server['status']}{NC}"
        )
        print(f"  Found: {BOLD}{server['model']}{NC} on :{server['port']} ({status})")
    else:
        # Multiple servers — show selection screen
        if _HAS_TERMIOS:
            _hide_cursor()
            try:
                server = _server_selection_screen(servers)
            finally:
                _show_cursor()
            if server is None:
                print("Aborted.")
                sys.exit(0)
        else:
            server = servers[0]
            print(f"  Using first: {server['model']} on :{server['port']}")

    chat_loop(server, args.temperature, args.max_tokens, args.system)


if __name__ == "__main__":
    main()
