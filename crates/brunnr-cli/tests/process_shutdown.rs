// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::{
    fs,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use brunnr_core::{
    AgentBinding, BrunnrConfig, CoordinationConfig, MemoryBackendKind, MemoryConfig, Mode, Role,
};
use brunnr_test_support::TempDir;
use thingr::{FilesTaskStore, NewTask, TaskStore};

#[tokio::test]
async fn sigterm_to_orchestrator_kills_tracked_worker_process_group() {
    let tempdir = TempDir::new("cli-sigterm-process");
    let task_root = tempdir.join("tasks");
    let memory_root = tempdir.join("memory");
    let registry = tempdir.join("spawns");
    let parent_pid_file = tempdir.join("worker.pid");
    let child_pid_file = tempdir.join("grandchild.pid");
    fs::create_dir_all(&memory_root).expect("memory root should be created");
    let store = FilesTaskStore::new(&task_root);
    let mut task = NewTask::primitive("Hold a worker process");
    task.id = Some("sigterm-task".to_string());
    task.description =
        "The worker intentionally sleeps until the orchestrator is terminated.".to_string();
    store.create(task).await.expect("task should be created");
    let script = format!(
        "echo $$ > \"{}\"; sleep 30 & echo $! > \"{}\"; wait",
        parent_pid_file.display(),
        child_pid_file.display()
    );
    let config = BrunnrConfig {
        mode: Mode::Orchestrate,
        memory: MemoryConfig {
            backend: MemoryBackendKind::Files,
            root: memory_root.display().to_string(),
            collection: "process-shutdown".to_string(),
            qdrant_url: None,
            qdrant_rest_url: None,
            qdrant_api_key_env: None,
            local_rerank_enabled: true,
            hyde_enabled: false,
            multi_query_enabled: false,
            debate_enabled: false,
            llm_consolidation_enabled: false,
        },
        agents: vec![AgentBinding {
            role: Role::Worker,
            agent: "sh".to_string(),
            model: None,
            command: Some("sh".to_string()),
            args: vec!["-c".to_string(), script],
            timeout_seconds: Some(30),
        }],
        coordination: CoordinationConfig {
            concurrency_limit: Some(1),
            max_retries: Some(0),
            spawn_registry_path: Some(registry.display().to_string()),
            max_concurrent_spawns: Some(2),
            spawn_max_lifetime_seconds: Some(30),
            spawn_shutdown_grace_millis: Some(50),
            ..CoordinationConfig::default()
        },
    };
    let config_path = tempdir.join("brunnr.toml");
    fs::write(
        &config_path,
        config.to_toml().expect("config should encode"),
    )
    .expect("config should write");
    let mut brunnr = Command::new(env!("CARGO_BIN_EXE_brunnr"))
        .arg("run")
        .arg("--config")
        .arg(&config_path)
        .arg("--root")
        .arg(&task_root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("brunnr should start");
    wait_for_file(&child_pid_file);
    let worker_pid = read_pid(&parent_pid_file);
    let grandchild_pid = read_pid(&child_pid_file);

    send_sigterm(brunnr.id());
    wait_for_exit(&mut brunnr);

    assert_pid_gone(worker_pid);
    assert_pid_gone(grandchild_pid);
}

fn send_sigterm(pid: u32) {
    let status = Command::new("kill")
        .arg("-TERM")
        .arg(pid.to_string())
        .status()
        .expect("kill should run");
    assert!(status.success(), "SIGTERM should be delivered");
}

fn wait_for_exit(child: &mut Child) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if child
            .try_wait()
            .expect("child status should be readable")
            .is_some()
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    panic!("brunnr did not exit after SIGTERM");
}

fn wait_for_file(path: &std::path::Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if path.exists() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("{} was not written", path.display());
}

fn read_pid(path: &std::path::Path) -> u32 {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .trim()
        .parse()
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn assert_pid_gone(pid: u32) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("pid {pid} survived orchestrator SIGTERM cleanup");
}

fn pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}
