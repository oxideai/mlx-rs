#!/usr/bin/env python3
"""Build and execute upstream sources outside Cargo's native build."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def run(build_root, reports):
    plan = json.loads((ROOT / "xtask/native-canaries/plan.json").read_text())
    native = ROOT / "mlx-sys/src/mlx-c"
    version = re.search(r"GIT_TAG v([^\)\s]+)", (native / "CMakeLists.txt").read_text())[1]
    identity = {"source_commit": output("git", "rev-parse", "HEAD"),
                "mlx_c_commit": output("git", "rev-parse", "HEAD", cwd=native),
                "nested_mlx_version": version,
                "cargo_lock_sha256": hashlib.sha256((ROOT / "Cargo.lock").read_bytes()).hexdigest()}
    dirty = bool(output("git", "status", "--porcelain", "--untracked-files=normal"))
    reports.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)
    source = build_root / "mlx-source"
    if source.exists():
        raise ValueError(f"native build must start empty: {source}")
    subprocess.run(["git", "clone", "--depth", "1", "--branch", f"v{version}",
                    "https://github.com/ml-explore/mlx.git", str(source)], check=True)
    mlx_commit = output("git", "rev-parse", "HEAD", cwd=source)
    failed = False
    for config in plan["configurations"]:
        started = time.monotonic()
        build = build_root / config["id"]
        build.mkdir()
        report = {**identity, "schema_version": 1, "configuration_id": config["id"],
                  "dirty": dirty, "mlx_commit": mlx_commit, "configuration": config,
                  "macos": output("sw_vers", "-productVersion"),
                  "sdk": output("xcrun", "--sdk", "macosx", "--show-sdk-version"),
                  "compiler": output("/usr/bin/c++", "--version"),
                  "cmake": output("cmake", "--version"),
                  "github_sha": os.environ.get("GITHUB_SHA"),
                  "github_pr_head_sha": os.environ.get("GITHUB_PR_HEAD_SHA"),
                  "commands": [], "runs": [], "verdict": "fail"}
        log_path = reports / (config["id"] + ".log")
        with log_path.open("w") as log:
            def execute(command, cwd=build, env=None):
                begin = time.monotonic()
                result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=log)
                record = {"command": list(map(str, command)), "exit_code": result.returncode,
                          "elapsed_seconds": time.monotonic() - begin}
                report["commands"].append(record)
                if result.returncode:
                    raise RuntimeError(f"command failed: {command}")
                return record

            try:
                execute(["cmake", "-S", str(ROOT / "xtask/native-canaries"), "-B", str(build),
                         f"-DMLX_C_SOURCE={native}", f"-DFETCHCONTENT_SOURCE_DIR_MLX={source}",
                         "-DCMAKE_C_COMPILER=/usr/bin/cc", "-DCMAKE_CXX_COMPILER=/usr/bin/c++",
                         f'-DCMAKE_BUILD_TYPE={config["profile"]}',
                         f'-DMLX_BUILD_METAL={config["metal"]}',
                         f'-DMLX_BUILD_ACCELERATE={config["accelerate"]}',
                         "-DMLX_METAL_JIT=OFF", f"-DMLX_METAL_PATH={build / 'metallib'}"])
                execute(["cmake", "--build", str(build), "--parallel", "4"])
                report["build_seconds"] = time.monotonic() - started
                report["native"] = dict(re.findall(r"^([^/#:\n]+):[^=\n]+=(.*)$",
                                                      (build / "CMakeCache.txt").read_text(), re.M))
                shutil.copyfile(ROOT / "conformance/fixtures/arithmetic.safetensors", build / "arrays.safetensors")
                metal = config["metal"] == "ON"
                for example in plan["examples"] + (plan["metal_examples"] if metal else []):
                    record = execute([str(build / "mlxc/examples" / example)])
                    report["runs"].append({**record, "id": example, "passed": 1})
                for index, test in enumerate(plan["tests"] + (plan["metal_tests"] if metal else [])):
                    xml = reports / f'{config["id"]}-{index}.xml'
                    record = execute([str(build / "upstream-tests/tests"), f"--test-case={test}",
                                      "--case-sensitive=true", "--reporters=xml", f"--out={xml}"],
                                     env={**os.environ, "DEVICE": "gpu" if metal else "cpu"})
                    result = ET.parse(xml).getroot()
                    cases = result.findall(".//TestCase")
                    counts = result.find("OverallResultsTestCases")
                    assertions = result.find("OverallResultsAsserts")
                    if (len(cases) != 1 or cases[0].get("name") != test
                            or counts is None or counts.get("successes") != "1"
                            or counts.get("failures") != "0" or assertions is None
                            or int(assertions.get("successes", "0")) == 0
                            or assertions.get("failures") != "0"):
                        raise ValueError(f"missing, skipped or failed C++ case: {test}")
                    report["runs"].append({**record, "id": test, "passed": 1,
                                           "assertions": int(assertions.get("successes"))})
                if output("git", "status", "--porcelain", "--untracked-files=normal"):
                    raise ValueError("dirty checkout")
                if output("git", "rev-parse", "HEAD") != identity["source_commit"]:
                    raise ValueError("source changed during run")
                report["verdict"] = "pass"
            except Exception as error:
                report["error"] = str(error)
                failed = True
        report["elapsed_seconds"] = time.monotonic() - started
        (reports / (config["id"] + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        print(f'{config["id"]}: {report["verdict"]}, {report["elapsed_seconds"]:.1f}s', flush=True)
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.build_root.resolve(), args.reports.resolve()))
