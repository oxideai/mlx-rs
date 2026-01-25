use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn get_repo_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).parent().unwrap().to_path_buf()
}

fn get_current_tag(mlx_c_dir: &Path) -> String {
    let output = Command::new("git")
        .args(["describe", "--tags"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to get current tag");

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn get_latest_tag(mlx_c_dir: &Path) -> String {
    // Fetch tags first
    Command::new("git")
        .args(["fetch", "--tags", "--quiet"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to fetch tags");

    let output = Command::new("git")
        .args(["rev-list", "--tags", "--max-count=1"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to get latest tag commit");

    let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();

    let output = Command::new("git")
        .args(["describe", "--tags", &commit])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to get tag name");

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn checkout_tag(mlx_c_dir: &Path, tag: &str) {
    Command::new("git")
        .args(["checkout", tag, "--quiet"])
        .current_dir(mlx_c_dir)
        .output()
        .expect("Failed to checkout tag");
}

fn generate_bindings(root_dir: &Path) -> String {
    let mlx_c_dir = root_dir.join("mlx-sys/src/mlx-c");

    let bindings = bindgen::Builder::default()
        .header(mlx_c_dir.join("mlx/c/mlx.h").to_str().unwrap())
        .header(mlx_c_dir.join("mlx/c/linalg.h").to_str().unwrap())
        .header(mlx_c_dir.join("mlx/c/error.h").to_str().unwrap())
        .header(mlx_c_dir.join("mlx/c/transforms_impl.h").to_str().unwrap())
        .clang_arg(format!("-I{}", mlx_c_dir.to_str().unwrap()))
        .generate()
        .expect("Unable to generate bindings");

    bindings.to_string()
}

fn print_diff(old: &str, new: &str, current_tag: &str, target_tag: &str, root_dir: &Path) {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    // Use a simple set-based diff for overview
    use std::collections::HashSet;
    let old_set: HashSet<&str> = old_lines.iter().copied().collect();
    let new_set: HashSet<&str> = new_lines.iter().copied().collect();

    let removed: Vec<&str> = old_lines
        .iter()
        .filter(|l| !new_set.contains(*l))
        .copied()
        .collect();
    let added: Vec<&str> = new_lines
        .iter()
        .filter(|l| !old_set.contains(*l))
        .copied()
        .collect();

    let additions = added.len();
    let removals = removed.len();

    // Build output for both console and file
    let mut output = String::new();
    output.push_str(&format!(
        "=== Bindings Diff ({} -> {}) ===\n\n",
        current_tag, target_tag
    ));
    output.push_str(&format!("+{} / -{} lines changed\n\n", additions, removals));

    if !removed.is_empty() {
        output.push_str("=== Removed ===\n");
        for line in &removed {
            output.push_str(&format!("- {}\n", line));
        }
        output.push('\n');
    }

    if !added.is_empty() {
        output.push_str("=== Added ===\n");
        for line in &added {
            output.push_str(&format!("+ {}\n", line));
        }
    }

    // Print to console (with colors, truncated)
    println!(
        "\n\x1b[32m+{}\x1b[0m / \x1b[31m-{}\x1b[0m lines changed\n",
        additions, removals
    );

    if !removed.is_empty() {
        println!("\x1b[31m=== Removed ===\x1b[0m");
        for line in removed.iter().take(50) {
            println!("\x1b[31m- {}\x1b[0m", line);
        }
        if removed.len() > 50 {
            println!("... and {} more removed lines", removed.len() - 50);
        }
        println!();
    }

    if !added.is_empty() {
        println!("\x1b[32m=== Added ===\x1b[0m");
        for line in added.iter().take(100) {
            println!("\x1b[32m+ {}\x1b[0m", line);
        }
        if added.len() > 100 {
            println!("... and {} more added lines", added.len() - 100);
        }
    }

    // Save full diff to file
    let diff_file = root_dir.join(format!("mlx-c-diff-{}-to-{}.txt", current_tag, target_tag));
    if let Err(e) = std::fs::write(&diff_file, &output) {
        eprintln!("Warning: Failed to write diff file: {}", e);
    } else {
        println!(
            "\n\x1b[33mFull diff saved to:\x1b[0m {}",
            diff_file.display()
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_tag = args.get(1).cloned();

    let root_dir = get_repo_root();
    let mlx_c_dir = root_dir.join("mlx-sys/src/mlx-c");

    println!("\x1b[33mChecking for mlx-c updates...\x1b[0m\n");

    let current_tag = get_current_tag(&mlx_c_dir);
    let latest_tag = get_latest_tag(&mlx_c_dir);
    let target_tag = target_tag.unwrap_or(latest_tag.clone());

    println!("Current version: \x1b[32m{}\x1b[0m", current_tag);
    println!("Latest version:  \x1b[32m{}\x1b[0m", latest_tag);
    if target_tag != latest_tag {
        println!("Target version:  \x1b[32m{}\x1b[0m", target_tag);
    }
    println!();

    if current_tag == target_tag {
        println!("\x1b[32mAlready up to date!\x1b[0m");
        return;
    }

    println!("\x1b[33mGenerating bindings for {}...\x1b[0m", current_tag);
    let current_bindings = generate_bindings(&root_dir);

    println!("\x1b[33mChecking out {}...\x1b[0m", target_tag);
    checkout_tag(&mlx_c_dir, &target_tag);

    println!("\x1b[33mGenerating bindings for {}...\x1b[0m", target_tag);
    let target_bindings = generate_bindings(&root_dir);

    // Restore original
    println!("\x1b[33mRestoring {}...\x1b[0m", current_tag);
    checkout_tag(&mlx_c_dir, &current_tag);

    println!(
        "\n\x1b[33m=== Bindings Diff ({} -> {}) ===\x1b[0m",
        current_tag, target_tag
    );

    if current_bindings == target_bindings {
        println!("\n\x1b[32mNo API changes detected!\x1b[0m");
    } else {
        print_diff(
            &current_bindings,
            &target_bindings,
            &current_tag,
            &target_tag,
            &root_dir,
        );
    }

    println!("\n\x1b[33mTo update, run:\x1b[0m");
    println!("  cd mlx-sys/src/mlx-c && git checkout {}", target_tag);
}
