// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use artesian_test_support::TempDir;

#[test]
fn cli_memory_mode_round_trip_and_spawn_alias_work() {
    let tempdir = TempDir::new("cli");
    let home = tempdir.join("home");
    let binary = env!("CARGO_BIN_EXE_artesian");

    let init = Command::new(binary)
        .arg("init")
        .env("ARTESIAN_HOME", &home)
        .current_dir(tempdir.path())
        .output()
        .expect("init should run");
    assert!(init.status.success(), "{}", stderr(&init));
    assert!(std::fs::read_to_string(tempdir.join(".mcp.json"))
        .expect("Claude MCP config should be written")
        .contains("artesian-mcp"));
    assert!(
        std::fs::read_to_string(home.join(".codex").join("config.toml"))
            .expect("Codex config should be written")
            .contains("artesian-mcp")
    );
    assert!(
        std::fs::read_to_string(home.join(".config").join("zed").join("settings.json"))
            .expect("Zed settings should be written")
            .contains("artesian-mcp")
    );

    let spawn = Command::new(binary)
        .args(["spawn", "worker", "echo", "--arg", "artesian-spawn"])
        .current_dir(tempdir.path())
        .output()
        .expect("spawn should run");
    assert!(spawn.status.success(), "{}", stderr(&spawn));
    assert!(stdout(&spawn).contains("role=worker agent=echo"));
    assert!(stdout(&spawn).contains("artesian-spawn"));

    let store = Command::new(binary)
        .args([
            "memory",
            "store",
            "Artesian memory mode works",
            "--tag",
            "smoke",
            "--node-id",
            "node:cli",
            "--source",
            "cli-test",
            "--confidence",
            "0.75",
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
    let find_out = stdout(&find);
    assert!(find_out.contains("node:cli\tArtesian memory mode works"));
    assert!(find_out.contains("source=cli-test"));
    assert!(find_out.contains("confidence=0.75"));

    let answer = Command::new(binary)
        .args([
            "memory",
            "answer",
            "What memory mode works?",
            "--limit",
            "1",
        ])
        .current_dir(tempdir.path())
        .output()
        .expect("answer should run");
    assert!(answer.status.success(), "{}", stderr(&answer));
    let answer_json: serde_json::Value =
        serde_json::from_str(&stdout(&answer)).expect("answer should be JSON");
    assert_eq!(answer_json["extractive"], true);
    assert_eq!(answer_json["sources"][0], "node:cli");
    assert!(answer_json["answer"]
        .as_str()
        .expect("answer should be a string")
        .contains("[node:cli]"));

    let commit = Command::new(binary)
        .args(["memory", "commit", "memory works", "--budget-tokens", "256"])
        .current_dir(tempdir.path())
        .output()
        .expect("commit should run");
    assert!(commit.status.success(), "{}", stderr(&commit));
    let commit_out = stdout(&commit);
    assert!(
        commit_out.contains("Artesian memory mode works"),
        "{commit_out}"
    );
    assert!(commit_out.contains("\"admitted\": 1"), "{commit_out}");

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

#[test]
fn cli_backfill_reports_bad_markdown_and_imports_tasks() {
    let tempdir = TempDir::new("cli-import");
    let binary = env!("CARGO_BIN_EXE_artesian");
    let import_dir = tempdir.join("import");
    std::fs::create_dir_all(import_dir.join("tasks/todo")).expect("import dirs should exist");
    std::fs::write(
        import_dir.join("memory.md"),
        "# Memory\n\nDurable import memory",
    )
    .expect("memory should be written");
    std::fs::write(
        import_dir.join("tasks/todo/task-one.md"),
        "# Imported Task\n\nTask body",
    )
    .expect("task should be written");
    std::fs::write(import_dir.join("broken.md"), [0xff, 0xfe])
        .expect("broken markdown should be written");

    let backfill = Command::new(binary)
        .args(["backfill", import_dir.to_str().expect("utf8 path")])
        .current_dir(tempdir.path())
        .output()
        .expect("backfill should run");
    assert!(backfill.status.success(), "{}", stderr(&backfill));
    assert!(stdout(&backfill).contains("failed=1"));
    assert!(stdout(&backfill).contains("task_imported"));

    let list = Command::new(binary)
        .args(["task", "list"])
        .current_dir(tempdir.path())
        .output()
        .expect("task list should run");
    assert!(list.status.success(), "{}", stderr(&list));
    assert!(stdout(&list).contains("Imported Task"));
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
