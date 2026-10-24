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


def case_assertions(result, test):
    cases = [case for case in result.findall(".//TestCase") if case.get("skipped") != "true"]
    counts = result.find("OverallResultsTestCases")
    assertions = result.find("OverallResultsAsserts")
    if (len(cases) != 1 or cases[0].get("name") != test
            or counts is None or counts.get("successes") != "1"
            or counts.get("failures") != "0" or assertions is None
            or int(assertions.get("successes", "0")) == 0
            or assertions.get("failures") != "0"):
        raise ValueError(f"missing, skipped or failed C++ case: {test}")
    return int(assertions.get("successes"))


def retain_admission_files(build, report):
    exports = report.get("linked_exports", {})
    keep = {build / "CMakeCache.txt", build / "linked-archives.txt"}
    keep.update(Path(archive["path"]) for archive in exports.get("archives", []))
    if exports.get("canary", {}).get("path"):
        keep.add(Path(exports["canary"]["path"]))
    for directory, _, files in os.walk(build, topdown=False):
        directory = Path(directory)
        for name in files:
            path = directory / name
            if path not in keep:
                path.unlink()
        if directory != build and not any(directory.iterdir()):
            directory.rmdir()


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
    if not source.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", f"v{version}",
                        "https://github.com/ml-explore/mlx.git", str(source)], check=True)
    if (output("git", "describe", "--tags", "--exact-match", cwd=source) != f"v{version}"
            or output("git", "status", "--porcelain", cwd=source)):
        raise ValueError("native MLX source must be clean at the pinned tag")
    mlx_commit = output("git", "rev-parse", "HEAD", cwd=source)
    failed = False
    for config in plan["configurations"]:
        subprocess.run(["df", "-h", str(build_root)], check=True)
        started = time.monotonic()
        build = build_root / config["id"]
        build.mkdir(exist_ok=True)
        (build / "metallib").mkdir(exist_ok=True)
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
                compile_commands = json.loads((build / "compile_commands.json").read_text())
                array_compile = next(c["command"] for c in compile_commands
                                     if c["file"] == str(source / "mlx/array.cpp"))
                report["native"]["MLX_BUILD_ACCELERATE"] = (
                    "ON" if "-DMLX_USE_ACCELERATE" in array_compile else "OFF")
                if report["native"]["MLX_BUILD_ACCELERATE"] != config["accelerate"]:
                    raise ValueError("compiled Accelerate backend differs from plan")
                report["array_compile_command"] = array_compile
                exports_path = reports / "exports" / (config["id"] + ".json")
                exports_path.parent.mkdir(exist_ok=True)
                export_command = ["cargo", "xtask", "linked-exports",
                                  "--build", str(build), "--report", str(exports_path)]
                export_result = subprocess.run(export_command, cwd=ROOT, stdout=log, stderr=log)
                report["linked_exports"] = json.loads(exports_path.read_text())
                report["linked_exports"]["command"] = export_command
                report["linked_exports"]["exit_code"] = export_result.returncode
                shutil.copyfile(ROOT / "conformance/fixtures/fast.safetensors", build / "arrays.safetensors")
                metal = config["metal"] == "ON"
                for example in plan["examples"] + (plan["metal_examples"] if metal else []):
                    record = execute([str(build / "mlxc/examples" / example)])
                    report["runs"].append({**record, "id": example, "passed": 1})
                for index, test in enumerate(plan["tests"] + (plan["metal_tests"] if metal else [])):
                    xml = reports / f'{config["id"]}-{index}.xml'
                    xml.unlink(missing_ok=True)
                    record = execute([str(build / "upstream-tests/tests"), f"--test-case={test}",
                                      "--case-sensitive=true", "--reporters=xml", f"--out={xml}"],
                                     env={**os.environ, "DEVICE": "gpu" if metal else "cpu"})
                    result = ET.parse(xml).getroot()
                    report["runs"].append({**record, "id": test, "passed": 1,
                                           "assertions": case_assertions(result, test)})
                if (output("git", "status", "--porcelain", "--untracked-files=normal")
                        or output("git", "status", "--porcelain", cwd=source)):
                    raise ValueError("dirty checkout")
                if output("git", "rev-parse", "HEAD") != identity["source_commit"]:
                    raise ValueError("source changed during run")
                if report["linked_exports"]["exit_code"]:
                    raise ValueError("linked export admission failed; see exports report")
                report["verdict"] = "pass"
            except Exception as error:
                report["error"] = str(error)
                failed = True
        report["elapsed_seconds"] = time.monotonic() - started
        (reports / (config["id"] + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        retain_admission_files(build, report)
        if report["verdict"] == "pass":
            subprocess.run(["cargo", "xtask", "linked-exports", "--verify",
                            str(exports_path)], cwd=ROOT, check=True)
        subprocess.run(["df", "-h", str(build_root)], check=True)
        print(f'{config["id"]}: {report["verdict"]}, {report["elapsed_seconds"]:.1f}s', flush=True)
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.build_root.resolve(), args.reports.resolve()))
