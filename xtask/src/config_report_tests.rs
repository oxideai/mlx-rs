use super::*;

fn fixture() -> (Vec<Configuration>, Vec<Value>, Value) {
    let configs = configurations(
        &serde_json::from_str(include_str!("../../ledger/supported-feature-matrix.json")).unwrap(),
    )
    .unwrap();
    let identity = json!({"source_commit": "checkout", "mlx_c_commit": "native", "nested_mlx_version": "0.32.2", "cargo_lock_sha256": "lock"});
    let reports =
        configs
            .iter()
            .map(|c| {
                let mut r = identity.clone();
                r.as_object_mut().unwrap().extend(json!({
            "schema_version": 1, "configuration_id": c.id, "dirty": false,
            "profile": c.profile, "toolchain": c.toolchain,
            "target_triple": c.target_triple, "command": test_command(c),
            "rustc": if c.toolchain == "stable" { "rustc 1.98.0" } else { "rustc 1.88.0" },
            "macos": "26.0", "sdk": "26.0", "resolved_features": c.resolved_features,
            "native": c.native, "verdict": "pass", "exit_code": 0,
            "tests": [{"id": "demo::lib::demo", "passed": 2, "failed": 0, "ignored": 0,
                "filtered_out": 0, "summaries": 1, "not_run": false, "outcome": "pass"}]
        }).as_object().unwrap().clone());
                r["native"]["mlx_version"] = "0.32.2".into();
                r["native"]["mlx_commit"] = "nested-commit".into();
                r
            })
            .collect();
    (configs, reports, identity)
}

fn check(configs: &[Configuration], reports: &[Value], identity: &Value) -> Result<Value, String> {
    validate_set(configs, reports, identity, |_, _| {
        Ok(BTreeSet::from(["demo::lib::demo".into()]))
    })
}

fn rejects(change: impl FnOnce(&mut Value), expected: &str) {
    let (configs, mut reports, identity) = fixture();
    change(&mut reports[0]);
    assert!(check(&configs, &reports, &identity)
        .unwrap_err()
        .contains(expected));
}

#[test]
fn admits_exact_required_set() {
    let (configs, reports, identity) = fixture();
    assert_eq!(
        check(&configs, &reports, &identity).unwrap()["verdict"],
        "pass"
    );
}

#[test]
fn rejects_missing_report() {
    let (configs, mut reports, identity) = fixture();
    reports.pop();
    assert!(check(&configs, &reports, &identity)
        .unwrap_err()
        .contains("missing required"));
}

#[test]
fn rejects_duplicate_report() {
    let (configs, mut reports, identity) = fixture();
    reports.push(reports[0].clone());
    assert!(check(&configs, &reports, &identity)
        .unwrap_err()
        .contains("duplicate configuration"));
}

#[test]
fn rejects_stale_checkout_even_if_pr_head_matches() {
    rejects(
        |r| {
            r["github_pr_head_sha"] = "checkout".into();
            r["source_commit"] = "old".into();
        },
        "source_commit",
    );
}

#[test]
fn rejects_failed_report() {
    rejects(|r| r["verdict"] = "fail".into(), "failed or skipped");
}

#[test]
fn rejects_skipped_report() {
    rejects(|r| r["verdict"] = "skipped".into(), "failed or skipped");
}

#[test]
fn rejects_nonzero_command_exit() {
    rejects(|r| r["exit_code"] = 101.into(), "failed or skipped");
}

#[test]
fn rejects_wrong_resolved_features() {
    rejects(
        |r| r["resolved_features"]["mlx-sys"] = json!(["accelerate", "metal"]),
        "features mismatch",
    );
}

#[test]
fn rejects_wrong_native_setting() {
    rejects(
        |r| r["native"]["MLX_BUILD_METAL"] = "ON".into(),
        "native setting",
    );
}

#[test]
fn rejects_not_run_binary() {
    rejects(
        |r| r["tests"][0]["not_run"] = true.into(),
        "skipped or incomplete",
    );
}

#[test]
fn rejects_zero_test_binary() {
    rejects(|r| r["tests"][0]["passed"] = 0.into(), "zero tests");
}

#[test]
fn rejects_missing_binary() {
    rejects(|r| r["tests"] = json!([]), "inventory differs");
}

#[test]
fn rejects_duplicate_binary() {
    rejects(
        |r| {
            let run = r["tests"][0].clone();
            r["tests"].as_array_mut().unwrap().push(run);
        },
        "duplicate test",
    );
}

#[test]
fn rejects_failed_binary_even_if_report_claims_pass() {
    rejects(|r| r["tests"][0]["failed"] = 1.into(), "failed, skipped");
}

#[test]
fn rejects_incomplete_binary() {
    rejects(|r| r["tests"][0]["summaries"] = 0.into(), "incomplete");
}

#[test]
fn rejects_filtered_tests() {
    rejects(|r| r["tests"][0]["filtered_out"] = 1.into(), "skipped");
}

#[test]
fn rejects_wrong_environment_and_command() {
    for field in [
        "mlx_c_commit",
        "nested_mlx_version",
        "cargo_lock_sha256",
        "profile",
        "toolchain",
        "target_triple",
        "command",
        "rustc",
        "macos",
        "sdk",
    ] {
        let (configs, mut reports, identity) = fixture();
        reports[0][field] = Value::Null;
        assert!(check(&configs, &reports, &identity).is_err(), "{field}");
    }
    rejects(|r| r["dirty"] = true.into(), "dirty");
}

#[test]
fn rejects_wrong_msrv_and_unstable_toolchains() {
    let (configs, mut reports, identity) = fixture();
    reports[2]["rustc"] = "rustc 1.98.0".into();
    assert!(check(&configs, &reports, &identity)
        .unwrap_err()
        .contains("rustc version"));
    rejects(|r| r["rustc"] = "rustc 1.99.0-nightly".into(), "not stable");
}

#[test]
fn additional_configuration_needs_no_count_gate_change() {
    let (mut configs, mut reports, identity) = fixture();
    let mut config = configs[0].clone();
    config.id = "future-configuration".into();
    let mut report = reports[0].clone();
    report["configuration_id"] = config.id.clone().into();
    configs.push(config);
    reports.push(report);
    check(&configs, &reports, &identity).unwrap();
}

#[test]
fn rejects_not_run_and_zero_doctests() {
    let (configs, mut reports, identity) = fixture();
    for report in &mut reports {
        report["tests"][0]["id"] = "demo::doctest::demo".into();
    }
    let expected =
        |_: &Configuration, _: &Value| Ok(BTreeSet::from(["demo::doctest::demo".into()]));
    validate_set(&configs, &reports, &identity, expected).unwrap();
    reports[0]["tests"][0]["not_run"] = true.into();
    assert!(validate_set(&configs, &reports, &identity, expected).is_err());
    reports[0]["tests"][0]["not_run"] = false.into();
    reports[0]["tests"][0]["passed"] = 0.into();
    assert!(validate_set(&configs, &reports, &identity, expected).is_err());
}

#[test]
fn matrix_rejects_invalid_shapes() {
    let matrix: Value =
        serde_json::from_str(include_str!("../../ledger/supported-feature-matrix.json")).unwrap();
    for field in [
        "id",
        "required",
        "profile",
        "toolchain",
        "native",
        "resolved_features",
        "packages",
        "allow_empty",
    ] {
        let mut broken = matrix.clone();
        broken["builds"][0].as_object_mut().unwrap().remove(field);
        assert!(configurations(&broken).is_err(), "{field}");
    }
    let mut broken = matrix.clone();
    broken["builds"][1]["id"] = broken["builds"][0]["id"].clone();
    assert!(configurations(&broken).is_err());
    let mut broken = matrix;
    for b in broken["builds"].as_array_mut().unwrap() {
        b["required"] = false.into();
    }
    assert!(configurations(&broken).is_err());
}

#[test]
fn rejects_stale_built_nested_mlx() {
    rejects(
        |r| r["native"]["mlx_version"] = "0.30.6".into(),
        "nested MLX identity",
    );
}

#[test]
fn command_pins_compilation_target() {
    let (configs, _, _) = fixture();
    let command = test_command(&configs[0]);
    assert!(command
        .windows(2)
        .any(|p| p == ["--target", "aarch64-apple-darwin"]));
}

#[test]
fn rejects_jit_cache_for_non_jit_configuration() {
    rejects(
        |r| r["native"]["MLX_METAL_JIT"] = "ON".into(),
        "native setting",
    );
}

#[test]
fn detects_json_not_run_diagnostic() {
    let runs = parse_runs("Doc-tests demo\n{\"status\":\"NOT RUN\"}\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s").unwrap();
    assert_eq!(runs[0]["not_run"], true);
}
