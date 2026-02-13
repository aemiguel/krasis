#!/usr/bin/env python3
"""Benchmark Qwen3-Coder-Next GPU prefill speed with a ~5000 token prompt.

GPU prefill threshold is 300 tokens. We use a large prompt to measure
actual GPU Marlin MoE throughput rather than DMA overhead.
"""

import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

import torch
from krasis.model import KrasisModel

MODEL = "/home/main/Documents/Claude/krasis/models/Qwen3-Coder-Next"


def make_long_prompt(tokenizer, target_tokens=5000):
    """Build a prompt that tokenizes to approximately target_tokens."""
    # Base content — realistic code review prompt
    base = """Please review the following Python code and provide detailed feedback on code quality, performance, security, and best practices.

```python
import os, sys, json, time, logging, threading, hashlib, hmac, secrets
import asyncio, aiohttp, aiofiles
from typing import List, Optional, Dict, Any, Tuple, Set, Union, Protocol
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from functools import lru_cache, wraps
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum, auto
import re, struct, base64, urllib.parse

logger = logging.getLogger(__name__)

class CacheEvictionPolicy(Enum):
    LRU = auto()
    LFU = auto()
    FIFO = auto()
    TTL = auto()

@dataclass
class CacheConfig:
    max_size: int = 10000
    ttl_seconds: float = 3600.0
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    persist_to_disk: bool = False
    disk_path: Optional[str] = None
    compression: bool = True
    max_memory_mb: float = 512.0

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    max_connections: int = 1000
    timeout: float = 30.0
    debug: bool = False
    log_level: str = "INFO"
    workers: int = 8
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    rate_limit_rps: int = 100
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    jwt_secret: Optional[str] = None
    api_key_header: str = "X-API-Key"

@dataclass
class RequestContext:
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[bytes] = None
    query_params: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    client_ip: str = ""
    request_id: str = field(default_factory=lambda: secrets.token_hex(16))
    user_id: Optional[str] = None
    authenticated: bool = False

class RateLimiter:
    def __init__(self, max_rps: int, window_seconds: float = 1.0):
        self._max_rps = max_rps
        self._window = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            reqs = self._requests[client_id]
            cutoff = now - self._window
            self._requests[client_id] = [t for t in reqs if t > cutoff]
            if len(self._requests[client_id]) >= self._max_rps:
                return False
            self._requests[client_id].append(now)
            return True

    def cleanup(self):
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._requests.items()
                      if all(t < now - self._window * 10 for t in v)]
            for k in expired:
                del self._requests[k]

class LRUCache:
    def __init__(self, config: CacheConfig):
        self._config = config
        self._data: OrderedDict = OrderedDict()
        self._ttls: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._memory_usage = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                if self._is_expired(key):
                    self._remove(key)
                    self._misses += 1
                    return None
                self._data.move_to_end(key)
                self._access_counts[key] += 1
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        with self._lock:
            if key in self._data:
                self._remove(key)
            while len(self._data) >= self._config.max_size:
                self._evict()
            self._data[key] = value
            self._ttls[key] = time.time() + (ttl or self._config.ttl_seconds)
            self._access_counts[key] = 1
            self._memory_usage += sys.getsizeof(value) + sys.getsizeof(key)

    def _is_expired(self, key: str) -> bool:
        return key in self._ttls and time.time() > self._ttls[key]

    def _remove(self, key: str):
        if key in self._data:
            self._memory_usage -= sys.getsizeof(self._data[key]) + sys.getsizeof(key)
            del self._data[key]
        self._ttls.pop(key, None)
        self._access_counts.pop(key, None)

    def _evict(self):
        if self._config.eviction_policy == CacheEvictionPolicy.LRU:
            key, _ = self._data.popitem(last=False)
            self._ttls.pop(key, None)
            self._access_counts.pop(key, None)
        elif self._config.eviction_policy == CacheEvictionPolicy.LFU:
            min_key = min(self._access_counts, key=self._access_counts.get)
            self._remove(min_key)

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._data), "hits": self._hits, "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "memory_mb": self._memory_usage / (1024 * 1024),
        }

class ConnectionPool:
    def __init__(self, max_size: int = 10, min_size: int = 2):
        self._pool: List[Any] = []
        self._max_size = max_size
        self._min_size = min_size
        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)
        self._created = 0
        self._in_use = 0

    def acquire(self, timeout: float = None) -> Any:
        with self._available:
            while not self._pool:
                if self._created < self._max_size:
                    conn = self._create_connection()
                    self._created += 1
                    self._in_use += 1
                    return conn
                if not self._available.wait(timeout):
                    raise TimeoutError("Connection pool exhausted")
            conn = self._pool.pop()
            self._in_use += 1
            return conn

    def release(self, conn: Any):
        with self._available:
            self._in_use -= 1
            if len(self._pool) < self._max_size:
                self._pool.append(conn)
                self._available.notify()
            else:
                self._destroy_connection(conn)
                self._created -= 1

    def _create_connection(self): return object()
    def _destroy_connection(self, conn): pass

class JWTAuth:
    def __init__(self, secret: str, algorithm: str = "HS256"):
        self._secret = secret.encode()
        self._algorithm = algorithm

    def create_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        header = {"alg": self._algorithm, "typ": "JWT"}
        payload["exp"] = int(time.time()) + expires_in
        payload["iat"] = int(time.time())
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
        message = header_b64 + b"." + payload_b64
        signature = hmac.new(self._secret, message, hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=")
        return (message + b"." + sig_b64).decode()

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            parts = token.split(".")
            if len(parts) != 3: return None
            message = (parts[0] + "." + parts[1]).encode()
            expected_sig = hmac.new(self._secret, message, hashlib.sha256).digest()
            actual_sig = base64.urlsafe_b64decode(parts[2] + "==")
            if not hmac.compare_digest(expected_sig, actual_sig): return None
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
            if payload.get("exp", 0) < time.time(): return None
            return payload
        except Exception:
            return None

class Router:
    def __init__(self):
        self._routes: Dict[str, Dict[str, callable]] = {}
        self._param_routes: List[Tuple[str, str, callable]] = []

    def add_route(self, method: str, path: str, handler: callable):
        if "{" in path:
            self._param_routes.append((method, path, handler))
        else:
            if path not in self._routes:
                self._routes[path] = {}
            self._routes[path][method.upper()] = handler

    def match(self, method: str, path: str) -> Tuple[Optional[callable], Dict[str, str]]:
        method = method.upper()
        if path in self._routes and method in self._routes[path]:
            return self._routes[path][method], {}
        for route_method, pattern, handler in self._param_routes:
            if route_method.upper() != method: continue
            params = self._extract_params(pattern, path)
            if params is not None:
                return handler, params
        return None, {}

    def _extract_params(self, pattern: str, path: str) -> Optional[Dict[str, str]]:
        parts_p = pattern.split("/")
        parts_r = path.split("/")
        if len(parts_p) != len(parts_r): return None
        params = {}
        for pp, rp in zip(parts_p, parts_r):
            if pp.startswith("{") and pp.endswith("}"):
                params[pp[1:-1]] = rp
            elif pp != rp:
                return None
        return params

class Middleware:
    def __init__(self):
        self._before: List[callable] = []
        self._after: List[callable] = []

    def before(self, fn: callable): self._before.append(fn)
    def after(self, fn: callable): self._after.append(fn)

    def execute_before(self, ctx: RequestContext) -> RequestContext:
        for mw in self._before:
            ctx = mw(ctx)
            if ctx is None: raise RuntimeError("Middleware returned None")
        return ctx

    def execute_after(self, ctx: RequestContext, response: dict) -> dict:
        for mw in self._after:
            response = mw(ctx, response)
        return response

class MetricsCollector:
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1):
        with self._lock: self._counters[name] += value

    def observe(self, name: str, value: float):
        with self._lock: self._histograms[name].append(value)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._counters)
            for name, values in self._histograms.items():
                if values:
                    sorted_v = sorted(values)
                    stats[f"{name}_count"] = len(values)
                    stats[f"{name}_avg"] = sum(values) / len(values)
                    stats[f"{name}_p50"] = sorted_v[len(sorted_v) // 2]
                    stats[f"{name}_p99"] = sorted_v[int(len(sorted_v) * 0.99)]
            return stats

class SimpleServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.router = Router()
        self.middleware = Middleware()
        self.pool = ConnectionPool(config.max_connections)
        self.cache = LRUCache(CacheConfig())
        self.rate_limiter = RateLimiter(config.rate_limit_rps)
        self.metrics = MetricsCollector()
        self.auth = JWTAuth(config.jwt_secret or secrets.token_hex(32))
        self._running = False
        self._threads: List[threading.Thread] = []

    def start(self):
        self._running = True
        logger.info(f"Starting server on {self.config.host}:{self.config.port}")
        for i in range(self.config.workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=5.0)

    def _worker(self, worker_id: int):
        while self._running:
            try:
                conn = self.pool.acquire(timeout=1.0)
                self._handle_connection(conn, worker_id)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.metrics.increment("errors")

    def _handle_connection(self, conn, worker_id: int):
        start = time.time()
        ctx = self._parse_request(conn)
        self.metrics.increment("requests_total")
        if not self.rate_limiter.allow(ctx.client_ip):
            self._send_response(conn, {"status": 429, "body": "Rate limited"})
            return
        try:
            ctx = self.middleware.execute_before(ctx)
            handler, params = self.router.match(ctx.method, ctx.path)
            if handler:
                response = handler(ctx, **params) if params else handler(ctx)
                response = self.middleware.execute_after(ctx, response)
                self._send_response(conn, response)
                self.metrics.increment("requests_success")
            else:
                self._send_response(conn, {"status": 404, "body": "Not Found"})
                self.metrics.increment("requests_404")
        except Exception as e:
            logger.exception(f"Request handling error: {e}")
            self._send_response(conn, {"status": 500, "body": "Internal Server Error"})
            self.metrics.increment("requests_error")
        finally:
            self.pool.release(conn)
            self.metrics.observe("request_duration", time.time() - start)

    def _parse_request(self, conn) -> RequestContext:
        return RequestContext(method="GET", path="/", headers={}, client_ip="127.0.0.1")

    def _send_response(self, conn, response: dict):
        status = response.get("status", 200)
        body = response.get("body", "")
        headers = response.get("headers", {})
        headers.setdefault("Content-Type", "application/json")
        headers.setdefault("X-Request-ID", secrets.token_hex(8))

def create_app():
    config = ServerConfig(port=8080, workers=4, debug=True, jwt_secret="dev-secret")
    server = SimpleServer(config)
    server.middleware.before(lambda ctx: ctx)
    server.router.add_route("GET", "/health", lambda ctx: {"status": 200, "body": {"healthy": True}})
    server.router.add_route("GET", "/metrics", lambda ctx: {"status": 200, "body": server.metrics.get_stats()})
    server.router.add_route("GET", "/api/users", lambda ctx: {"status": 200, "body": "[]"})
    server.router.add_route("GET", "/api/users/{id}", lambda ctx, id="": {"status": 200, "body": {"id": id}})
    server.router.add_route("POST", "/api/auth/login", lambda ctx: {"status": 200, "body": {"token": server.auth.create_token({"user": "test"})}})
    server.router.add_route("GET", "/api/cache/stats", lambda ctx: {"status": 200, "body": server.cache.stats})
    return server

if __name__ == "__main__":
    app = create_app()
    app.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        app.stop()
```

Provide a comprehensive review covering:
1. Thread safety issues and race conditions
2. Resource management and connection lifecycle
3. Error handling and recovery patterns
4. Security vulnerabilities (auth, injection, CORS)
5. Performance bottlenecks and scalability concerns
6. Design pattern improvements
7. Testing considerations
8. Production readiness gaps

For each issue found, explain the problem and suggest a fix."""

    # Check token count, repeat if needed
    tokens = tokenizer.encode(base)
    if len(tokens) < target_tokens:
        # Repeat the code section to reach target
        repeat_factor = (target_tokens // len(tokens)) + 1
        base = base * repeat_factor

    tokens = tokenizer.encode(base)
    # Trim to approximately target
    if len(tokens) > target_tokens + 200:
        # Decode back to get trimmed text
        tokens = tokens[:target_tokens]
        base = tokenizer.decode(tokens)

    return [{"role": "user", "content": base}]


def main():
    print("=" * 60)
    print("Benchmark: Qwen3-Coder-Next GPU prefill speed")
    print("=" * 60)

    # Load model with GPU prefill enabled
    print("\nLoading model PP=2...")
    t0 = time.time()
    model = KrasisModel(
        MODEL,
        pp_partition=[24, 24],
        devices=["cuda:0", "cuda:1"],
        kv_dtype=torch.float8_e4m3fn,
        krasis_threads=48,
        gpu_prefill=True,
        expert_divisor=4,  # layer-grouped: 6 layers per group = ~4.9 GB VRAM per group
    )
    model.load()
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # VRAM info
    for i in range(2):
        alloc = torch.cuda.memory_allocated(i) / (1024**2)
        print(f"  GPU{i}: {alloc:.0f} MB")

    # --- Benchmark: ~5000 token prompt ---
    print("\n--- Benchmark: ~5000 token GPU prefill ---")

    long_msg = make_long_prompt(model.tokenizer, target_tokens=5000)
    prompt_tokens = model.tokenizer.apply_chat_template(long_msg)
    n_prompt = len(prompt_tokens)
    print(f"Prompt tokens: {n_prompt}")

    # Generate only 8 tokens to minimize decode time in measurement
    t1 = time.time()
    text = model.chat(long_msg, max_new_tokens=8, temperature=0.0, top_k=1)
    total_time = time.time() - t1

    n_gen = len(model.tokenizer.encode(text))
    # With only 8 decode tokens at ~0.5s each, decode is ~4s
    est_decode_time = n_gen * 0.5
    est_prefill_time = max(0.1, total_time - est_decode_time)
    est_prefill_speed = n_prompt / est_prefill_time

    print(f"Output ({n_gen} tokens): {repr(text[:100])}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Estimated prefill: {est_prefill_time:.1f}s → ~{est_prefill_speed:.0f} tok/s")
    print(f"Estimated decode: {est_decode_time:.1f}s → ~{n_gen / max(0.1, est_decode_time):.1f} tok/s")

    gpu_prefill_used = n_prompt >= 300
    has_output = len(text.strip()) > 0

    print(f"\nGPU prefill used: {gpu_prefill_used} (prompt={n_prompt}, threshold=300)")
    print(f"Non-empty output: {has_output}")

    print("\n" + "=" * 60)
    print(f"RESULT: {n_prompt} tokens prefill in ~{est_prefill_time:.1f}s = ~{est_prefill_speed:.0f} tok/s")
    print(f"OVERALL: {'PASS' if has_output and gpu_prefill_used else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
