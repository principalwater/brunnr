// SPDX-License-Identifier: Apache-2.0

use std::{process::Command, sync::Arc};

use aquifer::{FilesBackend, SessionKey, SessionStore};
use artesian_test_support::TempDir;
use headgate::{LifecycleEntry, SnapshotEntry, WorkingContextBundle, WorkingContextSnapshot};

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

#[test]
fn cli_memory_find_expand_includes_relation_neighbor() {
    let tempdir = TempDir::new("cli-expand");
    let binary = env!("CARGO_BIN_EXE_artesian");
    let import_dir = tempdir.join("import");
    std::fs::create_dir_all(&import_dir).expect("import dir should exist");
    std::fs::write(
        import_dir.join("anchor.json"),
        serde_json::to_string(&serde_json::json!({
            "content": "needle relation anchor",
            "tier": "l1-atom",
            "node_id": "node:anchor",
            "relations": [{
                "subject": "AnchorMemory",
                "predicate": "links",
                "object": "SharedEntity",
                "source_node_id": ""
            }]
        }))
        .expect("anchor JSON should serialize"),
    )
    .expect("anchor should be written");
    std::fs::write(
        import_dir.join("neighbor.json"),
        serde_json::to_string(&serde_json::json!({
            "content": "connected neighbor fact",
            "tier": "l1-atom",
            "node_id": "node:neighbor",
            "relations": [{
                "subject": "SharedEntity",
                "predicate": "explains",
                "object": "NeighborFact",
                "source_node_id": ""
            }]
        }))
        .expect("neighbor JSON should serialize"),
    )
    .expect("neighbor should be written");

    let backfill = Command::new(binary)
        .args(["backfill", import_dir.to_str().expect("utf8 path")])
        .current_dir(tempdir.path())
        .output()
        .expect("backfill should run");
    assert!(backfill.status.success(), "{}", stderr(&backfill));

    let default_find = Command::new(binary)
        .args(["memory", "find", "needle", "--limit", "1"])
        .current_dir(tempdir.path())
        .output()
        .expect("default find should run");
    assert!(default_find.status.success(), "{}", stderr(&default_find));
    let default_out = stdout(&default_find);
    assert!(default_out.contains("node:anchor"), "{default_out}");
    assert!(!default_out.contains("node:neighbor"), "{default_out}");

    let expanded_find = Command::new(binary)
        .args(["memory", "find", "needle", "--limit", "1", "--expand"])
        .current_dir(tempdir.path())
        .output()
        .expect("expanded find should run");
    assert!(expanded_find.status.success(), "{}", stderr(&expanded_find));
    let expanded_out = stdout(&expanded_find);
    assert!(expanded_out.contains("node:anchor"), "{expanded_out}");
    assert!(expanded_out.contains("node:neighbor"), "{expanded_out}");
}

#[test]
fn cli_handoff_and_session_list_read_committed_session() {
    let tempdir = TempDir::new("cli-session");
    let binary = env!("CARGO_BIN_EXE_artesian");
    let key = SessionKey::new(
        Some("user-a".to_string()),
        Some("session-cli".to_string()),
        Some("task-cli".to_string()),
    );
    let entries = vec![SnapshotEntry::now(
        "anchor-task",
        "task-state",
        "cli handoff restores this state",
        1.0,
    )];
    let token_count = entries.iter().map(|entry| entry.tokens).sum();
    let bundle = WorkingContextBundle::new(
        WorkingContextSnapshot {
            schema: vec!["task-state".to_string()],
            budget_tokens: 4096,
            token_count,
            entries,
        },
        vec![LifecycleEntry::commit("anchor-task")],
    );
    let session = bundle
        .to_ocf_session(&key, Some("codex".to_string()))
        .expect("session should serialize");
    tokio::runtime::Runtime::new()
        .expect("runtime should start")
        .block_on(async {
            SessionStore::new(Arc::new(FilesBackend::new(tempdir.join(".artesian"))))
                .store(session)
                .await
                .expect("session should store");
        });

    let handoff = Command::new(binary)
        .args([
            "handoff",
            "session-cli",
            "--user",
            "user-a",
            "--task",
            "task-cli",
        ])
        .current_dir(tempdir.path())
        .output()
        .expect("handoff should run");
    assert!(handoff.status.success(), "{}", stderr(&handoff));
    let packet: serde_json::Value =
        serde_json::from_str(&stdout(&handoff)).expect("handoff should print JSON");
    assert_eq!(packet["session"]["handed_off_from"], "codex");
    assert!(packet["restored_working_state"]
        .as_str()
        .expect("state should be text")
        .contains("cli handoff restores this state"));

    let list = Command::new(binary)
        .args(["session", "list", "--user", "user-a"])
        .current_dir(tempdir.path())
        .output()
        .expect("session list should run");
    assert!(list.status.success(), "{}", stderr(&list));
    let summaries: serde_json::Value =
        serde_json::from_str(&stdout(&list)).expect("session list should print JSON");
    assert_eq!(summaries[0]["key"]["session_id"], "session-cli");
}

#[test]
fn cli_loop_max_wall_secs_exits_nonzero() {
    let tempdir = TempDir::new("cli-loop-wall-cap");
    let home = tempdir.join("home");
    let runs = tempdir.join("runs");
    let binary = env!("CARGO_BIN_EXE_artesian");

    let output = Command::new(binary)
        .arg("loop")
        .arg("--goal")
        .arg("false")
        .arg("--poll")
        .arg("--max-turns")
        .arg("2")
        .arg("--max-wall-secs")
        .arg("0")
        .arg("--root")
        .arg(tempdir.join(".artesian"))
        .env("ARTESIAN_HOME", &home)
        .env("ARTESIAN_RUNS_DIR", &runs)
        .env("ARTESIAN_STOP_FILE", tempdir.join("STOP"))
        .current_dir(tempdir.path())
        .output()
        .expect("loop should run");

    assert!(!output.status.success(), "loop should fail on wall cap");
    assert!(stderr(&output).contains("max-wall-secs"));
    let logs: Vec<_> = std::fs::read_dir(&runs)
        .expect("run-log dir should exist")
        .collect();
    assert_eq!(logs.len(), 1);
}

#[test]
fn cli_loop_stop_sentinel_exits_nonzero() {
    let tempdir = TempDir::new("cli-loop-stop");
    let home = tempdir.join("home");
    let runs = tempdir.join("runs");
    let stop = tempdir.join("STOP");
    std::fs::write(&stop, "stop").expect("stop sentinel should be written");
    let binary = env!("CARGO_BIN_EXE_artesian");

    let output = Command::new(binary)
        .arg("loop")
        .arg("--goal")
        .arg("false")
        .arg("--poll")
        .arg("--max-turns")
        .arg("2")
        .arg("--root")
        .arg(tempdir.join(".artesian"))
        .env("ARTESIAN_HOME", &home)
        .env("ARTESIAN_RUNS_DIR", &runs)
        .env("ARTESIAN_STOP_FILE", &stop)
        .current_dir(tempdir.path())
        .output()
        .expect("loop should run");

    assert!(
        !output.status.success(),
        "loop should fail on STOP sentinel"
    );
    assert!(stderr(&output).contains("stopped by sentinel"));
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
