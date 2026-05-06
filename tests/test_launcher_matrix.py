import argparse
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from krasis import launcher as launcher_mod
from krasis.launcher import Launcher, LauncherConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
NONEXISTENT_MODEL = "/tmp/nonexistent-krasis-launcher-matrix-model"


def _parse_key_value_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@contextlib.contextmanager
def _patched_launcher_env():
    old_detect = launcher_mod.detect_hardware
    old_execvp = os.execvp
    old_krasis_home = os.environ.get("KRASIS_HOME")
    with tempfile.TemporaryDirectory(prefix="krasis-launcher-home-") as home:
        launcher_mod.detect_hardware = lambda: {
            "gpu_count": 2,
            "gpus": [
                {"index": 0, "name": "Test GPU 0", "memory_total_mb": 24_000},
                {"index": 1, "name": "Test GPU 1", "memory_total_mb": 12_000},
            ],
        }
        os.environ["KRASIS_HOME"] = home
        try:
            yield
        finally:
            launcher_mod.detect_hardware = old_detect
            os.execvp = old_execvp
            if old_krasis_home is None:
                os.environ.pop("KRASIS_HOME", None)
            else:
                os.environ["KRASIS_HOME"] = old_krasis_home


class _ExecIntercept(Exception):
    def __init__(self, path: str, args: list[str]):
        super().__init__(path, args)
        self.path = path
        self.args = args


def _capture_launch_config(cfg: LauncherConfig, *, benchmark: bool = True) -> tuple[Path, list[str]]:
    args = argparse.Namespace()
    with _patched_launcher_env():
        launcher = Launcher(args)
        launcher.cfg = cfg
        launcher.selected_gpus = [
            {"index": idx, "name": f"Test GPU {idx}", "memory_total_mb": 24_000}
            for idx in cfg.selected_gpu_indices
        ]
        if not launcher.selected_gpus:
            launcher.selected_gpus = [{"index": 0, "name": "Test GPU 0", "memory_total_mb": 24_000}]

        def fake_execvp(path: str, argv: list[str]) -> None:
            raise _ExecIntercept(path, list(argv))

        os.execvp = fake_execvp
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                launcher.launch_server(benchmark=benchmark)
        except _ExecIntercept as exc:
            config_path = Path(exc.args[exc.args.index("--config") + 1])
            return config_path, exc.args
    raise AssertionError("launcher.launch_server() did not exec")


def _base_config() -> LauncherConfig:
    cfg = LauncherConfig()
    cfg.model_path = NONEXISTENT_MODEL
    cfg.selected_gpu_indices = [0]
    cfg.host = "127.0.0.1"
    cfg.port = 65_501
    cfg.krasis_threads = 8
    return cfg


def _run_server_start_smoke(config_path: Path, scenario: str, expected_fragments: list[str]) -> str:
    with tempfile.TemporaryDirectory(prefix=f"krasis-{scenario}-run-") as run_dir:
        env = os.environ.copy()
        env["KRASIS_RUN_DIR"] = run_dir
        env["KRASIS_RUN_TYPE"] = f"launcher-matrix-{scenario}"
        python_path = str(REPO_ROOT / "python")
        env["PYTHONPATH"] = f"{python_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else python_path
        proc = subprocess.run(
            [sys.executable, "-m", "krasis.server", "--config", str(config_path), "--benchmark"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )
    output = proc.stdout
    if proc.returncode == 0:
        raise AssertionError(f"{scenario}: expected nonexistent model failure, got success\n{output}")
    forbidden = [
        "server.py: error: argument",
        "invalid float value",
        "invalid int value",
        "invalid choice",
        "usage: server.py",
    ]
    for text in forbidden:
        if text in output:
            raise AssertionError(f"{scenario}: server failed before launch on {text!r}\n{output}")
    required = [
        "=== Resolved arguments ===",
        f"model_path = '{NONEXISTENT_MODEL}'",
        "config.json",
    ] + expected_fragments
    for text in required:
        if text not in output:
            raise AssertionError(f"{scenario}: missing expected output {text!r}\n{output}")
    return output


class LauncherMatrixTest(unittest.TestCase):
    maxDiff = None

    def test_save_config_round_trip_keeps_advanced_fields(self) -> None:
        cfg = _base_config()
        cfg.expert_group_size = 64
        cfg.force_rebuild_cache = True
        cfg.force_rebuild_hqq_cache = True
        cfg.build_cache = True
        with tempfile.NamedTemporaryFile("w", suffix=".conf", prefix="krasis-save-roundtrip-", delete=False) as f:
            path = Path(f.name)
        try:
            launcher_mod._save_config(str(path), cfg.to_save_dict())
            values = _parse_key_value_config(path)
            self.assertEqual(values.get("CFG_EXPERT_GROUP_SIZE"), "64")
            self.assertEqual(values.get("CFG_FORCE_REBUILD_CACHE"), "1")
            self.assertEqual(values.get("CFG_FORCE_REBUILD_HQQ_CACHE"), "1")
            self.assertEqual(values.get("CFG_BUILD_CACHE"), "1")

            loaded = LauncherConfig()
            loaded.apply_saved(launcher_mod._load_config(str(path)))
            self.assertEqual(loaded.expert_group_size, 64)
            self.assertTrue(loaded.force_rebuild_cache)
            self.assertTrue(loaded.force_rebuild_hqq_cache)
            self.assertTrue(loaded.build_cache)
        finally:
            path.unlink(missing_ok=True)

    def test_launcher_generated_configs_start_server_parse_path(self) -> None:
        scenarios = []

        cfg = _base_config()
        cfg.attention_quant = "hqq6"
        cfg.hqq_auto_budget_pct = 50.0
        cfg.hqq46_auto_budget_mib = 256
        cfg.hqq_sidecar_manifest = ""
        scenarios.append((
            "plain_hqq6_stale_budget",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "hqq6",
                "CFG_KV_DTYPE": "k6v6",
                "CFG_GPU_EXPERT_BITS": "4",
                "CFG_CPU_EXPERT_BITS": "4",
                "CFG_SELECTED_GPUS": "0",
                "CFG_NUM_GPUS": "1",
            },
            ["attention_quant = 'hqq6'", "hqq_auto_budget_pct = None"],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq4"
        cfg.kv_dtype = "k4v4"
        cfg.hqq_auto_budget_pct = 20.0
        cfg.hqq46_auto_budget_mib = 128
        scenarios.append((
            "plain_hqq4_stale_budget",
            cfg,
            {"CFG_ATTENTION_QUANT": "hqq4", "CFG_KV_DTYPE": "k4v4"},
            ["attention_quant = 'hqq4'", "hqq_auto_budget_pct = None"],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq46_auto"
        cfg.hqq_auto_budget_pct = 10.0
        scenarios.append((
            "hqq46_auto_10pct",
            cfg,
            {"CFG_ATTENTION_QUANT": "hqq46_auto", "CFG_HQQ_AUTO_BUDGET_PCT": "10.0"},
            ["attention_quant = 'hqq46_auto'", "hqq_auto_budget_pct = 10.0"],
            {"CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq68_auto"
        cfg.hqq_auto_budget_pct = 10.0
        cfg.gpu_expert_bits = 8
        cfg.cpu_expert_bits = 8
        cfg.kv_dtype = "bf16"
        scenarios.append((
            "hqq68_auto_10pct_int8_bf16kv",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "hqq68_auto",
                "CFG_HQQ_AUTO_BUDGET_PCT": "10.0",
                "CFG_GPU_EXPERT_BITS": "8",
                "CFG_CPU_EXPERT_BITS": "8",
                "CFG_KV_DTYPE": "bf16",
            },
            ["attention_quant = 'hqq68_auto'", "hqq_auto_budget_pct = 10.0"],
            {"CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "bf16"
        cfg.kv_dtype = "k4v4"
        cfg.dynamic_hcs = False
        cfg.dynamic_hcs_tail_blocks = 5
        cfg.enable_thinking = False
        cfg.session_enabled = True
        cfg.force_load = True
        cfg.force_rebuild_cache = True
        cfg.force_rebuild_hqq_cache = True
        cfg.build_cache = True
        scenarios.append((
            "bf16_advanced_toggles",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "bf16",
                "CFG_DYNAMIC_HCS": "0",
                "CFG_DYNAMIC_HCS_TAIL_BLOCKS": "5",
                "CFG_ENABLE_THINKING": "0",
                "CFG_SESSION_ENABLED": "1",
                "CFG_FORCE_LOAD": "1",
                "CFG_FORCE_REBUILD_CACHE": "1",
                "CFG_FORCE_REBUILD_HQQ_CACHE": "1",
                "CFG_BUILD_CACHE": "1",
            },
            [
                "attention_quant = 'bf16'",
                "dynamic_hcs = False",
                "enable_thinking = False",
                "session_enabled = True",
            ],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        cfg = _base_config()
        cfg.attention_quant = "hqq8"
        cfg.selected_gpu_indices = [0, 1]
        cfg.layer_group_size = 6
        cfg.expert_group_size = 64
        cfg.vram_safety_margin = 900
        cfg.port = 65_502
        scenarios.append((
            "hqq8_two_gpu_shape",
            cfg,
            {
                "CFG_ATTENTION_QUANT": "hqq8",
                "CFG_SELECTED_GPUS": "0,1",
                "CFG_NUM_GPUS": "2",
                "CFG_LAYER_GROUP_SIZE": "6",
                "CFG_EXPERT_GROUP_SIZE": "64",
                "CFG_VRAM_SAFETY_MARGIN": "900",
                "CFG_PORT": "65502",
            },
            [
                "attention_quant = 'hqq8'",
                "num_gpus = 2",
                "selected_gpus = '0,1'",
                "expert_group_size = 64",
            ],
            {"CFG_HQQ_AUTO_BUDGET_PCT", "CFG_HQQ46_AUTO_BUDGET_MB", "CFG_HQQ_SIDECAR_MANIFEST"},
        ))

        generated_paths: list[Path] = []
        try:
            for name, cfg, expected, expected_output, absent_keys in scenarios:
                with self.subTest(name=name):
                    config_path, cmd_args = _capture_launch_config(cfg)
                    generated_paths.append(config_path)
                    values = _parse_key_value_config(config_path)
                    for key, value in expected.items():
                        self.assertEqual(values.get(key), value, f"{name}: {key}")
                    for key in absent_keys:
                        self.assertNotIn(key, values, f"{name}: inactive {key} should not be serialized")
                    self.assertIn("--config", cmd_args)
                    self.assertIn("--benchmark", cmd_args)
                    _run_server_start_smoke(config_path, name, expected_output)
        finally:
            for path in generated_paths:
                path.unlink(missing_ok=True)

    def test_legacy_blank_values_are_unset_before_argparse(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".conf", prefix="krasis-legacy-blank-", delete=False) as f:
            f.write(f'MODEL_PATH="{NONEXISTENT_MODEL}"\n')
            f.write('CFG_SELECTED_GPUS="0"\n')
            f.write('CFG_NUM_GPUS=""\n')
            f.write('CFG_KV_CACHE_MB=""\n')
            f.write('CFG_DYNAMIC_HCS_TAIL_BLOCKS=""\n')
            f.write('CFG_HQQ_AUTO_BUDGET_PCT=""\n')
            f.write('CFG_HQQ46_AUTO_BUDGET_MB=""\n')
            f.write('CFG_HQQ_SIDECAR_MANIFEST=""\n')
            f.write('CFG_GGUF_PATH=""\n')
            f.write('CFG_ATTENTION_QUANT="hqq6"\n')
            config_path = Path(f.name)
        try:
            _run_server_start_smoke(
                config_path,
                "legacy_blank_optional_values",
                [
                    "attention_quant = 'hqq6'",
                    "hqq_auto_budget_pct = None",
                    "hqq46_auto_budget_mib = None",
                    "dynamic_hcs_tail_blocks = 2",
                    "kv_cache_mb = 1000",
                    "num_gpus = 1",
                ],
            )
        finally:
            config_path.unlink(missing_ok=True)
