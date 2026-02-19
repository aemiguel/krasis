"""Benchmark suite runner — runs N models × M configs automatically.

Reads a TOML config defining models and hardware configurations, runs each
combination as a subprocess (python -m krasis.server --benchmark), captures
output, parses results, and writes a summary markdown table.

Usage:
    python -m krasis.suite                              # default config
    python -m krasis.suite --config path/to/suite.toml  # custom config
"""

import argparse
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ANSI formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


@dataclass
class SuiteCombo:
    """One model × config combination to benchmark."""
    model_name: str
    model_path: str
    num_gpus: int
    gpu_expert_bits: int = 4
    cpu_expert_bits: int = 4
    attention_quant: str = "int8"
    shared_expert_quant: str = "int8"
    dense_mlp_quant: str = "int8"
    lm_head_quant: str = "int8"
    kv_dtype: str = "fp8_e4m3"
    krasis_threads: int = 48
    gguf_path: str = ""

    @property
    def label(self) -> str:
        """Auto-generated label: {num_gpus}gpu_int{gpu}gpu_int{cpu}cpu."""
        return f"{self.num_gpus}gpu_int{self.gpu_expert_bits}gpu_int{self.cpu_expert_bits}cpu"

    @property
    def log_filename(self) -> str:
        """Filename for the log: {model}_{label}.log."""
        return f"{self.model_name}_{self.label}.log"


@dataclass
class SuiteResult:
    """Result from running one combo."""
    combo: SuiteCombo
    success: bool = False
    prefill_tok_s: float = 0.0
    prefill_ttft: float = 0.0
    decode_tok_s: float = 0.0
    decode_ms_per_tok: float = 0.0
    log_path: str = ""
    error: str = ""
    elapsed_s: float = 0.0


class SuiteRunner:
    """Orchestrates benchmark runs across model × config combinations."""

    TIMEOUT_S = 3600  # 1 hour per combo

    def __init__(self, config_path: str, output_dir: str = ""):
        self.config_path = config_path
        self.repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        if not output_dir:
            output_dir = os.path.join(self.repo_root, "benchmarks", "suite_logs")
        self.output_dir = output_dir
        self.krasis_home = os.environ.get(
            "KRASIS_HOME",
            os.path.join(os.path.expanduser("~"), ".krasis"),
        )
        self.models_dir = os.path.join(self.krasis_home, "models")
        self.python = self._find_python()

    def _find_python(self) -> str:
        """Find the correct python binary — prefer the repo venv."""
        venv_python = os.path.join(self.repo_root, ".venv", "bin", "python3")
        if os.path.isfile(venv_python):
            return venv_python
        return sys.executable

    def _find_gguf(self, name: str) -> str:
        """Find a GGUF by name under ~/.krasis/models/.

        The name can be:
        - A directory name (e.g. "DeepSeek-V2-Lite-GGUF") — finds the first
          .gguf file inside it (searching subdirectories too)
        - A .gguf filename (e.g. "model.Q4_K_M.gguf") — searches all subdirs

        Returns absolute path if found, empty string otherwise.
        """
        # Check if name is a directory under models/
        dir_path = os.path.join(self.models_dir, name)
        if os.path.isdir(dir_path):
            # Find first .gguf file (walk into subdirs like Q8_0/)
            for root, _dirs, files in os.walk(dir_path):
                for f in sorted(files):
                    if f.endswith(".gguf"):
                        return os.path.join(root, f)
            return ""

        # Otherwise search as a filename in subdirectories
        for direntry in os.scandir(self.models_dir):
            if not direntry.is_dir():
                continue
            candidate = os.path.join(direntry.path, name)
            if os.path.isfile(candidate):
                return candidate
        return ""

    def load_config(self) -> List[SuiteCombo]:
        """Parse TOML config and return all model × config combinations."""
        with open(self.config_path, "rb") as f:
            data = tomllib.load(f)

        configs = data.get("config", [])
        models = data.get("model", [])

        if not configs:
            raise ValueError(f"No [[config]] entries in {self.config_path}")
        if not models:
            raise ValueError(f"No [[model]] entries in {self.config_path}")

        combos = []
        for model in models:
            model_name = model["name"]
            # Resolve model path
            if "path" in model and model["path"]:
                model_path = model["path"]
            else:
                model_path = os.path.join(self.models_dir, model_name)

            if not os.path.isdir(model_path):
                print(f"{YELLOW}Warning: model path not found: {model_path}, skipping{NC}",
                      file=sys.stderr)
                continue

            gguf_path = model.get("gguf_path", "")
            gguf_name = model.get("gguf_name", "")
            if not gguf_path and gguf_name:
                gguf_path = self._find_gguf(gguf_name)
                if not gguf_path:
                    print(f"{YELLOW}Warning: GGUF '{gguf_name}' not found in {self.models_dir}, skipping{NC}",
                          file=sys.stderr)
                    continue

            for cfg in configs:
                combo = SuiteCombo(
                    model_name=model_name,
                    model_path=model_path,
                    num_gpus=cfg["num_gpus"],
                    gpu_expert_bits=cfg.get("gpu_expert_bits", 4),
                    cpu_expert_bits=cfg.get("cpu_expert_bits", 4),
                    attention_quant=cfg.get("attention_quant", "int8"),
                    shared_expert_quant=cfg.get("shared_expert_quant", "int8"),
                    dense_mlp_quant=cfg.get("dense_mlp_quant", "int8"),
                    lm_head_quant=cfg.get("lm_head_quant", "int8"),
                    kv_dtype=cfg.get("kv_dtype", "fp8_e4m3"),
                    krasis_threads=cfg.get("krasis_threads", 48),
                    gguf_path=gguf_path,
                )
                combos.append(combo)

        return combos

    def _build_cmd(self, combo: SuiteCombo) -> List[str]:
        """Build the subprocess command for one combo."""
        cmd = [
            self.python, "-m", "krasis.server",
            "--model-path", combo.model_path,
            "--num-gpus", str(combo.num_gpus),
            "--benchmark",
            "--kv-dtype", combo.kv_dtype,
            "--gpu-expert-bits", str(combo.gpu_expert_bits),
            "--cpu-expert-bits", str(combo.cpu_expert_bits),
            "--attention-quant", combo.attention_quant,
            "--shared-expert-quant", combo.shared_expert_quant,
            "--dense-mlp-quant", combo.dense_mlp_quant,
            "--lm-head-quant", combo.lm_head_quant,
            "--krasis-threads", str(combo.krasis_threads),
        ]
        if combo.gguf_path:
            cmd.extend(["--gguf-path", combo.gguf_path])
        return cmd

    def _build_env(self, combo: SuiteCombo) -> Dict[str, str]:
        """Build environment with CUDA_VISIBLE_DEVICES for the combo."""
        env = os.environ.copy()
        gpu_indices = ",".join(str(i) for i in range(combo.num_gpus))
        env["CUDA_VISIBLE_DEVICES"] = gpu_indices
        # Ensure instrumentation is off
        env.pop("KRASIS_DECODE_TIMING", None)
        return env

    def _parse_results(self, output: str) -> Dict[str, float]:
        """Parse benchmark output for prefill and decode averages.

        Expected formats:
          Prefill section:  Average: 1434.1 tok/s, TTFT=14.40s
          Decode section:   Average: 6.40 tok/s (156.1ms/tok)
        """
        results: Dict[str, float] = {}

        # Find prefill average: after "Prefill (" section header
        prefill_match = re.search(
            r"Prefill\s+\([^)]+\):\s*\n(?:.*\n)*?\s+Average:\s+([\d.]+)\s+tok/s,\s+TTFT=([\d.]+)s",
            output,
        )
        if prefill_match:
            results["prefill_tok_s"] = float(prefill_match.group(1))
            results["prefill_ttft"] = float(prefill_match.group(2))

        # Find decode average: after "Decode (" section header
        decode_match = re.search(
            r"Decode\s+\([^)]+\):\s*\n(?:.*\n)*?\s+Average:\s+([\d.]+)\s+tok/s\s+\(([\d.]+)ms/tok\)",
            output,
        )
        if decode_match:
            results["decode_tok_s"] = float(decode_match.group(1))
            results["decode_ms_per_tok"] = float(decode_match.group(2))

        return results

    def run_one(self, combo: SuiteCombo) -> SuiteResult:
        """Run a single benchmark combo as a subprocess."""
        result = SuiteResult(combo=combo)

        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, combo.log_filename)
        result.log_path = log_path

        cmd = self._build_cmd(combo)
        env = self._build_env(combo)

        print(f"  {DIM}$ CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
              f"{' '.join(cmd[1:])}{NC}")

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_S,
                env=env,
            )
            result.elapsed_s = time.time() - t0

            # Combine stdout + stderr for full log
            full_output = proc.stdout
            if proc.stderr:
                full_output += "\n\n--- STDERR ---\n" + proc.stderr

            # Write log
            with open(log_path, "w") as f:
                f.write(full_output)

            if proc.returncode != 0:
                result.error = f"Exit code {proc.returncode}"
                # Try to extract a useful error from the last few lines
                lines = proc.stderr.strip().split("\n") if proc.stderr else []
                if lines:
                    result.error += f": {lines[-1][:200]}"
                return result

            # Parse results from stdout
            parsed = self._parse_results(proc.stdout)
            if parsed:
                result.prefill_tok_s = parsed.get("prefill_tok_s", 0.0)
                result.prefill_ttft = parsed.get("prefill_ttft", 0.0)
                result.decode_tok_s = parsed.get("decode_tok_s", 0.0)
                result.decode_ms_per_tok = parsed.get("decode_ms_per_tok", 0.0)
                result.success = True
            else:
                result.error = "Could not parse benchmark results from output"

        except subprocess.TimeoutExpired:
            result.elapsed_s = time.time() - t0
            result.error = f"Timeout after {self.TIMEOUT_S}s"
            with open(log_path, "w") as f:
                f.write(f"TIMEOUT after {self.TIMEOUT_S}s\n")
        except Exception as e:
            result.elapsed_s = time.time() - t0
            result.error = str(e)
            with open(log_path, "w") as f:
                f.write(f"ERROR: {e}\n")

        return result

    def _clean_old_logs(self) -> int:
        """Delete old log and summary files from the output directory.

        Returns the number of files deleted.
        """
        if not os.path.isdir(self.output_dir):
            return 0
        count = 0
        for f in os.listdir(self.output_dir):
            if f.endswith(".log") or f.startswith("suite_summary_"):
                path = os.path.join(self.output_dir, f)
                try:
                    os.remove(path)
                    count += 1
                except OSError:
                    pass
        return count

    def run_all(self) -> List[SuiteResult]:
        """Run all combos sequentially, skipping on failure."""
        combos = self.load_config()
        if not combos:
            print(f"{RED}No valid model × config combinations found.{NC}")
            return []

        # Clean old logs before starting
        cleaned = self._clean_old_logs()
        if cleaned:
            print(f"{DIM}Cleaned {cleaned} old log/summary files from {self.output_dir}{NC}")

        total = len(combos)
        print(f"\n{BOLD}Krasis Benchmark Suite{NC}")
        print(f"  Config:  {self.config_path}")
        print(f"  Combos:  {total}")
        print(f"  Logs:    {self.output_dir}")
        print()

        results = []
        for i, combo in enumerate(combos):
            tag = f"[{i + 1}/{total}]"
            print(f"{BOLD}{tag} {combo.model_name} / {combo.label}{NC}")

            result = self.run_one(combo)
            results.append(result)

            if result.success:
                print(f"  {GREEN}Prefill: {result.prefill_tok_s:.1f} tok/s, "
                      f"TTFT={result.prefill_ttft:.2f}s{NC}")
                print(f"  {GREEN}Decode:  {result.decode_tok_s:.2f} tok/s "
                      f"({result.decode_ms_per_tok:.1f}ms/tok){NC}")
            else:
                print(f"  {RED}FAILED: {result.error}{NC}")

            print(f"  {DIM}Elapsed: {result.elapsed_s:.0f}s | Log: {result.log_path}{NC}")
            print()

        return results

    def write_summary(self, results: List[SuiteResult]) -> str:
        """Write a markdown summary table. Returns path to summary file."""
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, f"suite_summary_{timestamp}.md")

        lines = []
        lines.append(f"# Krasis Benchmark Suite — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(f"Config: `{self.config_path}`")
        lines.append("")

        # Summary table
        lines.append("| Model | Config | Prefill (tok/s) | TTFT (s) | Decode (tok/s) | ms/tok | Status | Log |")
        lines.append("|-------|--------|----------------:|----------:|---------------:|-------:|--------|-----|")

        for r in results:
            c = r.combo
            if r.success:
                status = "PASS"
                prefill = f"{r.prefill_tok_s:.1f}"
                ttft = f"{r.prefill_ttft:.2f}"
                decode = f"{r.decode_tok_s:.2f}"
                ms_tok = f"{r.decode_ms_per_tok:.1f}"
            else:
                status = f"FAIL"
                prefill = "-"
                ttft = "-"
                decode = "-"
                ms_tok = "-"

            log_rel = os.path.basename(r.log_path)
            lines.append(
                f"| {c.model_name} | {c.label} | {prefill} | {ttft} | {decode} | {ms_tok} | {status} | [{log_rel}]({log_rel}) |"
            )

        # Failures section
        failures = [r for r in results if not r.success]
        if failures:
            lines.append("")
            lines.append("## Failures")
            lines.append("")
            for r in failures:
                lines.append(f"- **{r.combo.model_name} / {r.combo.label}**: {r.error}")

        lines.append("")

        with open(summary_path, "w") as f:
            f.write("\n".join(lines))

        return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Krasis Benchmark Suite — run N models × M configs",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to benchmark_suite.toml (default: benchmarks/benchmark_suite.toml)",
    )
    parser.add_argument(
        "--output-dir", default="",
        help="Directory for logs and summary (default: benchmarks/suite_logs/)",
    )
    args = parser.parse_args()

    # Resolve default config path
    config_path = args.config
    if config_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        config_path = os.path.join(repo_root, "benchmarks", "benchmark_suite.toml")

    if not os.path.isfile(config_path):
        print(f"{RED}Config not found: {config_path}{NC}", file=sys.stderr)
        sys.exit(1)

    runner = SuiteRunner(config_path, output_dir=args.output_dir)
    results = runner.run_all()

    if results:
        summary_path = runner.write_summary(results)
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        print(f"{BOLD}{'═' * 48}")
        print(f"  SUITE COMPLETE: {passed} passed, {failed} failed")
        print(f"{'═' * 48}{NC}")
        print(f"  Summary: {summary_path}")
        print()


if __name__ == "__main__":
    main()
