// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use brunnr_test_support::TempDir;

#[test]
fn cli_memory_mode_round_trip_and_spawn_alias_work() {
    let tempdir = TempDir::new("cli");
    let home = tempdir.join("home");
    let binary = env!("CARGO_BIN_EXE_brunnr");

    let init = Command::new(binary)
        .arg("init")
        .env("BRUNNR_HOME", &home)
        .current_dir(tempdir.path())
        .output()
        .expect("init should run");
    assert!(init.status.success(), "{}", stderr(&init));
    assert!(std::fs::read_to_string(tempdir.join(".mcp.json"))
        .expect("Claude MCP config should be written")
        .contains("brunnr-mcp"));
    assert!(
        std::fs::read_to_string(home.join(".codex").join("config.toml"))
            .expect("Codex config should be written")
            .contains("brunnr-mcp")
    );
    assert!(
        std::fs::read_to_string(home.join(".config").join("zed").join("settings.json"))
            .expect("Zed settings should be written")
            .contains("brunnr-mcp")
    );

    let spawn = Command::new(binary)
        .args(["spawn", "thor", "codex"])
        .current_dir(tempdir.path())
        .output()
        .expect("spawn should run");
    assert!(spawn.status.success(), "{}", stderr(&spawn));
    assert!(stdout(&spawn).contains("role=worker alias=thor agent=codex"));

    let store = Command::new(binary)
        .args([
            "memory",
            "store",
            "Brunnr memory mode works",
            "--tag",
            "smoke",
            "--node-id",
            "node:cli",
        ])
        .current_dir(tempdir.path())
        .output()
        .expect("store should run");
    assert!(store.status.success(), "{}", stderr(&store));

    let find = Command::new(binary)
        .args(["memory", "find", "works", "--node-id", "node:cli"])
        .current_dir(tempdir.path())
        .output()
        .expect("find should run");
    assert!(find.status.success(), "{}", stderr(&find));
    assert!(stdout(&find).contains("node:cli\tBrunnr memory mode works"));

    let import_dir = tempdir.join("import");
    std::fs::create_dir_all(&import_dir).expect("import dir should be created");
    std::fs::write(
        import_dir.join("memory.md"),
        "[2026-01-02] CLI backfill is idempotent",
    )
    .expect("import file should be written");
    let backfill = Command::new(binary)
        .args(["backfill", import_dir.to_str().expect("utf8 path")])
        .current_dir(tempdir.path())
        .output()
        .expect("backfill should run");
    assert!(backfill.status.success(), "{}", stderr(&backfill));
    assert!(stdout(&backfill).contains("imported=1 skipped_duplicates=0"));
    let backfill_again = Command::new(binary)
        .args(["backfill", import_dir.to_str().expect("utf8 path")])
        .current_dir(tempdir.path())
        .output()
        .expect("second backfill should run");
    assert!(
        backfill_again.status.success(),
        "{}",
        stderr(&backfill_again)
    );
    assert!(stdout(&backfill_again).contains("imported=0 skipped_duplicates=1"));
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
