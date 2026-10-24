use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn hash(path: &Path) -> Result<String, String> {
    let out = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("cannot hash {}", path.display()));
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_owned)
        .ok_or("missing hash".into())
}

fn read(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn cases(oracle: &Value) -> Vec<String> {
    oracle["contract"]["cases"]
        .as_object()
        .unwrap()
        .keys()
        .flat_map(|name| {
            oracle["contract"]["devices"]
                .as_array()
                .unwrap()
                .iter()
                .map(move |device| format!("{name}/{}", device.as_str().unwrap()))
        })
        .collect()
}

pub(crate) fn checkpoint_evidence(root: &Path, log: &str) -> Result<Value, String> {
    let Some(path) = std::env::var_os("MLX_LM_REAL_CHECKPOINTS") else {
        return Ok(Value::Null);
    };
    collect_checkpoint_evidence(root, &PathBuf::from(path), Some(log))
}

fn collect_checkpoint_evidence(
    root: &Path,
    path: &Path,
    log: Option<&str>,
) -> Result<Value, String> {
    let manifest = read(path)?;
    let report_path = manifest["expected_report"]["path"]
        .as_str()
        .ok_or("missing oracle path")?;
    let mut report = read(Path::new(report_path))?;
    if manifest["expected_report"]["sha256"] != hash(Path::new(report_path))? {
        return Err("checkpoint oracle hash mismatch".into());
    }
    for key in [
        "schema_version",
        "entries",
        "cases",
        "devices",
        "python_environment",
    ] {
        if manifest[key] != report["contract"][key] {
            return Err("checkpoint contract mismatch".into());
        }
    }
    for (name, entry) in report["contract"]["entries"]
        .as_object_mut()
        .ok_or("missing entries")?
    {
        entry["local_directory"] = name.clone().into();
    }
    let oracle_path = root.join("conformance/mlx-lm/bump-oracle.json");
    let oracle = read(&oracle_path)?;
    validate_live_oracle(&report, &oracle)?;
    let receipt_path = Path::new(report_path).with_file_name("live-oracle-generation.json");
    let receipt = read(&receipt_path)?;
    for (key, file) in [
        ("generator_sha256", "live_oracle.py"),
        ("oracle_generator_sha256", "real_checkpoints.py"),
        ("requirements_sha256", "requirements.lock"),
        ("committed_sha256", "bump-oracle.json"),
    ] {
        if receipt[key] != hash(&root.join("conformance/mlx-lm").join(file))? {
            return Err(format!("live oracle provenance mismatch: {key}"));
        }
    }
    if receipt["python_version"] != "3.12.14"
        || receipt["live_sha256"] != manifest["expected_report"]["sha256"]
    {
        return Err("live oracle generation receipt mismatch".into());
    }
    let cases = cases(&oracle);
    for case in &cases {
        if log.is_some_and(|log| {
            log.lines()
                .filter(|line| line.contains(&format!("PASS: {case}:")))
                .count()
                != 1
        }) {
            return Err(format!("missing or duplicate checkpoint execution: {case}"));
        }
    }
    Ok(
        json!({"oracle_sha256": hash(&oracle_path)?, "manifest_path": path, "manifest_sha256": hash(path)?, "live_sha256": manifest["expected_report"]["sha256"], "generation": receipt, "cases": cases}),
    )
}

fn validate_live_oracle(report: &Value, oracle: &Value) -> Result<(), String> {
    if report["schema_version"] != oracle["schema_version"]
        || report["contract"] != oracle["contract"]
        || report["environment"] != oracle["environment"]
        || report["results"]
            .as_object()
            .map(|r| r.keys().collect::<Vec<_>>())
            != oracle["contract"]["cases"]
                .as_object()
                .map(|r| r.keys().collect::<Vec<_>>())
    {
        return Err("live oracle contract or environment mismatch".into());
    }
    for devices in report["results"]
        .as_object()
        .ok_or("missing live results")?
        .values()
    {
        if devices
            .as_object()
            .map(|d| d.keys().map(String::as_str).collect::<Vec<_>>())
            != Some(vec!["cpu", "metal"])
        {
            return Err("live oracle device inventory mismatch".into());
        }
        for device in ["cpu", "metal"] {
            let result = &devices[device];
            if result["completed"] != true
                || result["greedy_ids"].as_array().map(Vec::len) != Some(32)
                || result["greedy_steps"].as_array().map(Vec::len) != Some(32)
            {
                return Err("incomplete live oracle case".into());
            }
        }
    }
    Ok(())
}

fn validate_native(
    report: &Value,
    identity: &Value,
    config: &Value,
    plan: &Value,
) -> Result<(), String> {
    for key in [
        "source_commit",
        "mlx_c_commit",
        "nested_mlx_version",
        "cargo_lock_sha256",
    ] {
        if report[key] != identity[key] || report[key].as_str().is_none_or(str::is_empty) {
            return Err(format!("native: stale or missing {key}"));
        }
    }
    if report["schema_version"] != 1
        || report["dirty"] != false
        || report["verdict"] != "pass"
        || report["configuration"] != *config
    {
        return Err("native: failed, dirty or mismatched configuration".into());
    }
    for field in ["mlx_commit", "compiler", "cmake", "macos", "sdk"] {
        if report[field].as_str().is_none_or(str::is_empty) {
            return Err(format!("native: missing {field}"));
        }
    }
    for (key, value) in [
        ("CMAKE_BUILD_TYPE", &config["profile"]),
        ("MLX_BUILD_METAL", &config["metal"]),
        ("MLX_BUILD_ACCELERATE", &config["accelerate"]),
        ("MLX_METAL_JIT", &json!("OFF")),
        ("MLX_C_BUILD_EXAMPLES", &json!("ON")),
    ] {
        if report["native"][key] != *value {
            return Err(format!("native: wrong {key}"));
        }
    }
    if report["linked_exports"]["build_provenance"]["mlx_commit"] != report["mlx_commit"] {
        return Err("linked exports: different MLX build source".into());
    }
    if report["linked_exports"]["verdict"] != "pass" || report["linked_exports"]["exit_code"] != 0 {
        return Err("linked exports: failed execution".into());
    }
    for key in [
        "CMAKE_BUILD_TYPE",
        "MLX_BUILD_METAL",
        "MLX_BUILD_ACCELERATE",
        "MLX_METAL_JIT",
    ] {
        if report["linked_exports"]["configuration"][key] != report["native"][key] {
            return Err(format!("linked exports: mismatched {key}"));
        }
    }
    let mut expected = BTreeSet::new();
    for key in ["examples", "tests"]
        .into_iter()
        .chain(if config["metal"] == "ON" {
            vec!["metal_examples", "metal_tests"]
        } else {
            vec![]
        })
    {
        for id in plan[key].as_array().ok_or("invalid canary plan")? {
            expected.insert(id.as_str().ok_or("invalid canary id")?);
        }
    }
    let mut seen = BTreeSet::new();
    for run in report["runs"].as_array().ok_or("native: missing runs")? {
        let id = run["id"].as_str().ok_or("native: missing id")?;
        if !seen.insert(id)
            || run["exit_code"] != 0
            || run["passed"] != 1
            || (id.starts_with("test ") && run["assertions"].as_u64().unwrap_or(0) == 0)
        {
            return Err("native: duplicate, failed or empty run".into());
        }
    }
    if seen != expected {
        return Err("native: missing or unexpected canary".into());
    }
    let commands = report["commands"]
        .as_array()
        .ok_or("native: missing commands")?;
    if commands.len() != seen.len() + 2 || commands.iter().any(|c| c["exit_code"] != 0) {
        return Err("native: incomplete or failed commands".into());
    }
    Ok(())
}

pub(crate) fn validate(
    root: &Path,
    config_paths: &[PathBuf],
    native_paths: &[PathBuf],
) -> Result<Value, String> {
    let matrix = read(&root.join("ledger/supported-feature-matrix.json"))?;
    let configs: Vec<_> = crate::config_report::configurations(&matrix)?
        .into_iter()
        .filter(|c| c.required)
        .collect();
    let mut selected = Vec::new();
    let oracle = read(&root.join("conformance/mlx-lm/bump-oracle.json"))?;
    let oracle_hash = hash(&root.join("conformance/mlx-lm/bump-oracle.json"))?;
    let mut mlx_commit = None;
    for path in config_paths {
        let report = read(path)?;
        if !configs.iter().any(|c| report["configuration_id"] == c.id) {
            continue;
        }
        let config = configs
            .iter()
            .find(|c| report["configuration_id"] == c.id)
            .unwrap();
        if config.checkpoints_required {
            validate_checkpoints(&report, &oracle_hash, &cases(&oracle))?;
            let manifest = report["checkpoints"]["manifest_path"]
                .as_str()
                .ok_or("missing live manifest")?;
            if collect_checkpoint_evidence(root, Path::new(manifest), None)?
                != report["checkpoints"]
            {
                return Err("live checkpoint evidence changed after execution".into());
            }
        }
        if let Some(commit) = &mlx_commit {
            if commit != &report["native"]["mlx_commit"] {
                return Err("configurations built different MLX commits".into());
            }
        }
        mlx_commit = Some(report["native"]["mlx_commit"].clone());
        selected.push(path.clone());
    }
    crate::config_report::validate_reports(root, &selected, &configs)?;
    let identity = crate::config_report::identity(root)?;
    let plan = read(&root.join("xtask/native-canaries/plan.json"))?;
    let reports = native_paths
        .iter()
        .map(|path| read(path))
        .collect::<Result<Vec<_>, _>>()?;
    for report in &reports {
        crate::linked_exports::validate(root, &report["linked_exports"])?;
    }
    let seen = validate_native_set(
        &reports,
        &identity,
        &plan,
        mlx_commit.as_ref().ok_or("missing Rust MLX commit")?,
    )?;
    let exports: Vec<_> = reports
        .iter()
        .map(|report| {
            json!({
                "configuration_id": report["configuration_id"],
                "planned_count": report["linked_exports"]["planned_count"],
                "found_count": report["linked_exports"]["found_count"],
                "archives": report["linked_exports"]["archives"].as_array().unwrap().iter()
                    .map(|archive| json!({"path": archive["path"], "sha256": archive["sha256"]}))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    Ok(
        json!({"verdict": "pass", "source_commit": identity["source_commit"], "native_configurations": seen, "linked_exports": exports, "checkpoint_oracle_sha256": oracle_hash}),
    )
}

fn validate_native_set(
    reports: &[Value],
    identity: &Value,
    plan: &Value,
    mlx_commit: &Value,
) -> Result<BTreeSet<String>, String> {
    let mut seen = BTreeSet::new();
    for report in reports {
        let id = report["configuration_id"]
            .as_str()
            .ok_or("native: missing configuration id")?;
        if !seen.insert(id.to_owned()) {
            return Err("duplicate native configuration".into());
        }
        let config = plan["configurations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["id"] == id)
            .ok_or("unknown native configuration")?;
        validate_native(report, identity, config, plan)?;
        if &report["mlx_commit"] != mlx_commit {
            return Err("native and Rust builds used different MLX commits".into());
        }
    }
    if seen.len() != plan["configurations"].as_array().unwrap().len() {
        return Err("missing required native configuration".into());
    }
    Ok(seen)
}

fn validate_checkpoints(report: &Value, oracle_hash: &str, cases: &[String]) -> Result<(), String> {
    if report["checkpoints"]["oracle_sha256"] != oracle_hash
        || report["checkpoints"]["cases"] != json!(cases)
        || report["checkpoints"]["manifest_sha256"]
            .as_str()
            .is_none_or(|s| s.len() != 64)
    {
        return Err("missing, skipped or unpinned real-checkpoint evidence".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_oracle_requires_complete_results_for_the_input_contract() {
        let oracle: Value =
            serde_json::from_str(include_str!("../../conformance/mlx-lm/bump-oracle.json"))
                .unwrap();
        let mut live = oracle.clone();
        for name in oracle["contract"]["cases"].as_object().unwrap().keys() {
            for device in ["cpu", "metal"] {
                live["results"][name][device] = json!({
                    "completed": true,
                    "greedy_ids": vec![123; 32],
                    "greedy_steps": vec![json!({"top1_id": 123}); 32],
                });
            }
        }
        assert!(validate_live_oracle(&live, &oracle).is_ok());
        for pointer in [
            "/contract/cases",
            "/contract/entries",
            "/contract/devices",
            "/schema_version",
            "/environment",
            "/results",
            "/results/qwen3-06b-4bit/cpu/completed",
            "/results/qwen3-06b-4bit/cpu/greedy_ids",
            "/results/qwen3-06b-4bit/cpu/greedy_steps",
            "/results/qwen3-06b-4bit/metal",
        ] {
            let mut bad = live.clone();
            *bad.pointer_mut(pointer).unwrap() = Value::Null;
            assert!(validate_live_oracle(&bad, &oracle).is_err(), "{pointer}");
        }
        let mut missing = live.clone();
        missing["results"]
            .as_object_mut()
            .unwrap()
            .remove("qwen3-06b-4bit");
        assert!(validate_live_oracle(&missing, &oracle).is_err());
        let mut extra = live.clone();
        extra["results"]["extra"] = live["results"]["qwen3-06b-4bit"].clone();
        assert!(validate_live_oracle(&extra, &oracle).is_err());
        let mut extra_device = live.clone();
        extra_device["results"]["qwen3-06b-4bit"]["extra"] =
            live["results"]["qwen3-06b-4bit"]["cpu"].clone();
        assert!(validate_live_oracle(&extra_device, &oracle).is_err());
    }

    #[test]
    fn rejects_missing_skipped_or_changed_checkpoints() {
        let cases = vec!["model/cpu".to_owned(), "model/metal".to_owned()];
        let good = json!({"checkpoints": {"oracle_sha256": "pin", "manifest_sha256": "a".repeat(64), "cases": cases}});
        assert!(validate_checkpoints(&good, "pin", &cases).is_ok());
        for bad in [
            Value::Null,
            json!({"checkpoints": {"oracle_sha256": "pin", "cases": []}}),
        ] {
            assert!(validate_checkpoints(&bad, "pin", &cases).is_err());
        }
        assert!(validate_checkpoints(&good, "other-pin", &cases).is_err());
        let mut missing_case = good.clone();
        missing_case["checkpoints"]["cases"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(validate_checkpoints(&missing_case, "pin", &cases).is_err());
    }

    #[test]
    fn rejects_stale_failed_missing_duplicate_and_empty_native_runs() {
        let identity = json!({"source_commit": "sha", "mlx_c_commit": "c", "nested_mlx_version": "v", "cargo_lock_sha256": "lock"});
        let config = json!({"id": "cpu", "profile": "Debug", "metal": "OFF", "accelerate": "OFF"});
        let plan = json!({"examples": ["example"], "tests": ["test grad"]});
        let mut good = identity.clone();
        for (key, value) in json!({"schema_version": 1, "dirty": false, "verdict": "pass", "configuration": config, "mlx_commit": "mlx", "compiler": "clang", "cmake": "cmake", "macos": "mac", "sdk": "sdk", "native": {"CMAKE_BUILD_TYPE": "Debug", "MLX_BUILD_METAL": "OFF", "MLX_BUILD_ACCELERATE": "OFF", "MLX_METAL_JIT": "OFF", "MLX_C_BUILD_EXAMPLES": "ON"}, "runs": [{"id": "example", "exit_code": 0, "passed": 1}, {"id": "test grad", "exit_code": 0, "passed": 1, "assertions": 3}], "commands": [{"exit_code": 0}, {"exit_code": 0}, {"exit_code": 0}, {"exit_code": 0}]}).as_object().unwrap() { good[key] = value.clone(); }
        good["linked_exports"] = json!({"verdict": "pass", "exit_code": 0, "configuration": good["native"], "build_provenance": {"mlx_commit": "mlx"}});
        assert!(validate_native(&good, &identity, &config, &plan).is_ok());
        good["configuration_id"] = "cpu".into();
        let mut set_plan = plan.clone();
        set_plan["configurations"] = json!([config]);
        assert!(validate_native_set(&[good.clone()], &identity, &set_plan, &json!("mlx")).is_ok());
        for reports in [vec![], vec![good.clone(), good.clone()]] {
            assert!(validate_native_set(&reports, &identity, &set_plan, &json!("mlx")).is_err());
        }
        assert!(validate_native_set(
            &[good.clone()],
            &identity,
            &set_plan,
            &json!("different-mlx")
        )
        .is_err());
        let mut unknown = good.clone();
        unknown["configuration_id"] = "unknown".into();
        assert!(validate_native_set(&[unknown], &identity, &set_plan, &json!("mlx")).is_err());
        for (pointer, value) in [
            ("/linked_exports", Value::Null),
            ("/source_commit", json!("stale")),
            ("/verdict", json!("fail")),
            ("/dirty", json!(true)),
            ("/runs", json!([])),
            ("/runs/1/id", json!("example")),
            ("/runs/1/assertions", json!(0)),
            ("/runs/1/exit_code", json!(1)),
            ("/native/MLX_BUILD_METAL", json!("ON")),
        ] {
            let mut bad = good.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            assert!(
                validate_native(&bad, &identity, &config, &plan).is_err(),
                "{pointer}"
            );
        }
    }
}
