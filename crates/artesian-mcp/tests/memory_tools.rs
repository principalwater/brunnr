// SPDX-License-Identifier: Apache-2.0

use std::{
    fs,
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

use aquifer::{
    MemoryResult, SqliteVecVectorStore, TextEmbedder, VectorMemoryBackend, VectorMemoryConfig,
};
use artesian_core::{AgentBinding, AgentCatalog, AgentCatalogEntry, AgentModel, Mode, Role};
use artesian_mcp::{
    AnchorSetRequest, AnswerRequest, BindRequest, CommitRequest, DelegateRequest, FindRequest,
    LearnRequest, LoopRequest, MemoryServer, QualifyRequest, RelationRequest,
    SessionCheckpointRequest, SessionResumeByTaskRequest, SessionResumeRequest, SkillProcedureStep,
    SkillReplayRequest, SkillsRequest, StoreRequest, TeamCreateRequest, TeamMessageKindRequest,
    TeamMessageRequest, TeamRunRequest, TeamRunTaskRequest, TeamSpawnRequest, TeamStatusRequest,
    TeamTaskAddRequest, TeamTaskAwaitRequest, TeamTaskClaimRequest, TeamTaskCompleteRequest,
    ToolsFindRequest,
};
use artesian_test_support::TempDir;
use rmcp::handler::server::wrapper::Parameters;
use tokio_util::sync::CancellationToken;

type TestProgressCallback = Arc<dyn Fn(f64, Option<f64>, Option<String>) + Send + Sync>;

#[tokio::test]
async fn memory_tools_store_and_find_with_files_backend() {
    let tempdir = TempDir::new("mcp");
    let server = MemoryServer::new(tempdir.path());

    let stored = server
        .memory_store(Parameters(StoreRequest {
            content: "MCP memory tool round trip".to_string(),
            tags: Some(vec!["mcp".to_string()]),
            node_id: Some("node:mcp".to_string()),
            relations: None,
            source: Some("mcp-test".to_string()),
            confidence: Some(0.9),
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("store should succeed")
        .0;

    let found = server
        .memory_find(Parameters(FindRequest {
            query: "round".to_string(),
            limit: Some(5),
            node_id: Some("node:mcp".to_string()),
            expand: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("find should succeed")
        .0;

    assert_eq!(stored.node_id, "node:mcp");
    assert_eq!(found.hits.len(), 1);
    assert_eq!(found.hits[0].node_id, "node:mcp");
    assert_eq!(found.hits[0].content, "MCP memory tool round trip");
    assert_eq!(found.hits[0].source.as_deref(), Some("mcp-test"));
    assert_eq!(found.hits[0].confidence, Some(0.9));

    let answer = server
        .memory_answer(Parameters(AnswerRequest {
            question: "What round trip does MCP remember?".to_string(),
            limit: Some(1),
        }))
        .await
        .expect("answer should succeed")
        .0;
    assert!(answer.extractive);
    assert_eq!(answer.sources, vec!["node:mcp"]);
    assert!(answer.answer.contains("[node:mcp]"));
    assert!(answer.answer.contains("MCP memory tool round trip"));
}

#[tokio::test]
async fn memory_find_reports_project_scope_and_projects_tool_lists_values() {
    let tempdir = TempDir::new("mcp-projects");
    let server = MemoryServer::new(tempdir.path());

    for (node, project) in [
        ("node:mcp-project-a", "A"),
        ("node:mcp-project-shared", "shared"),
        ("node:mcp-project-b", "B"),
    ] {
        server
            .memory_store(Parameters(StoreRequest {
                content: format!("mcp partition sentinel {node}"),
                tags: None,
                node_id: Some(node.to_string()),
                relations: None,
                source: None,
                confidence: None,
                scope: None,
                agent_id: None,
                session_id: None,
                task_id: None,
                user_id: None,
                project: Some(project.to_string()),
            }))
            .await
            .expect("store should succeed");
    }

    let found = server
        .memory_find(Parameters(FindRequest {
            query: "mcp partition sentinel".to_string(),
            limit: Some(10),
            node_id: None,
            expand: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: Some("A".to_string()),
        }))
        .await
        .expect("find should succeed")
        .0;
    let nodes = found
        .hits
        .iter()
        .map(|hit| hit.node_id.as_str())
        .collect::<Vec<_>>();

    assert!(nodes.contains(&"node:mcp-project-a"), "{nodes:?}");
    assert!(nodes.contains(&"node:mcp-project-shared"), "{nodes:?}");
    assert!(!nodes.contains(&"node:mcp-project-b"), "{nodes:?}");
    assert_eq!(found.scope_applied.project, "A");
    assert_eq!(found.scope_applied.union, vec!["A", "shared", "(untagged)"]);

    let projects = server
        .memory_projects()
        .await
        .expect("projects should succeed")
        .0
        .projects;
    assert!(projects.contains(&"A".to_string()), "{projects:?}");
    assert!(projects.contains(&"B".to_string()), "{projects:?}");
    assert!(projects.contains(&"shared".to_string()), "{projects:?}");
}

#[tokio::test]
async fn memory_find_expand_includes_relation_neighbor() {
    let tempdir = TempDir::new("mcp-expand");
    let server = MemoryServer::new(tempdir.path());

    server
        .memory_store(Parameters(StoreRequest {
            content: "needle relation anchor".to_string(),
            tags: None,
            node_id: Some("node:anchor".to_string()),
            relations: Some(vec![RelationRequest {
                subject: "AnchorMemory".to_string(),
                predicate: "links".to_string(),
                object: "SharedEntity".to_string(),
                source_node_id: None,
            }]),
            source: None,
            confidence: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("store anchor should succeed");
    server
        .memory_store(Parameters(StoreRequest {
            content: "connected neighbor fact".to_string(),
            tags: None,
            node_id: Some("node:neighbor".to_string()),
            relations: Some(vec![RelationRequest {
                subject: "SharedEntity".to_string(),
                predicate: "explains".to_string(),
                object: "NeighborFact".to_string(),
                source_node_id: None,
            }]),
            source: None,
            confidence: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("store neighbor should succeed");

    let default_find = server
        .memory_find(Parameters(FindRequest {
            query: "needle".to_string(),
            limit: Some(1),
            node_id: None,
            expand: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("default find should succeed")
        .0;
    assert_eq!(default_find.hits.len(), 1);
    assert_eq!(default_find.hits[0].node_id, "node:anchor");

    let expanded_find = server
        .memory_find(Parameters(FindRequest {
            query: "needle".to_string(),
            limit: Some(1),
            node_id: None,
            expand: Some(true),
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("expanded find should succeed")
        .0;
    let nodes = expanded_find
        .hits
        .iter()
        .map(|hit| hit.node_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(nodes[0], "node:anchor");
    assert!(nodes.contains(&"node:neighbor"), "{nodes:?}");
}

#[tokio::test]
async fn memory_commit_runs_acc_cycle_with_files_backend() {
    let tempdir = TempDir::new("mcp-commit");
    let server = MemoryServer::new(tempdir.path());

    for content in [
        "the team chose Rust for the core crates",
        "deployments run nightly on kubernetes",
    ] {
        server
            .memory_store(Parameters(StoreRequest {
                content: content.to_string(),
                tags: None,
                node_id: None,
                relations: None,
                source: None,
                confidence: None,
                scope: None,
                agent_id: None,
                session_id: None,
                task_id: None,
                user_id: None,
                project: None,
            }))
            .await
            .expect("store should succeed");
    }

    let response = server
        .memory_commit(Parameters(CommitRequest {
            query: "rust deployment".to_string(),
            budget_tokens: Some(256),
            recall_limit: None,
            min_score: None,
        }))
        .await
        .expect("commit should succeed")
        .0;

    assert!(response.candidates >= 2);
    assert!(response.admitted >= 1);
    assert!(response.footprint_tokens <= response.budget_tokens);
    assert!(
        response.committed_context.contains("Rust")
            || response.committed_context.contains("deployment"),
        "{}",
        response.committed_context
    );
}

#[tokio::test]
async fn memory_qualify_returns_audited_admit_and_reject_decisions() {
    let tempdir = TempDir::new("mcp-qualify");
    let server = MemoryServer::new(tempdir.path());

    server
        .memory_store(Parameters(StoreRequest {
            content: "the team chose Rust for the core crates".to_string(),
            tags: None,
            node_id: None,
            relations: None,
            source: None,
            confidence: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("store should succeed");

    let admitted = server
        .memory_qualify(Parameters(QualifyRequest {
            candidate: "the deployment gate checks freshness".to_string(),
            goal: Some("deployment gate".to_string()),
        }))
        .await
        .expect("qualify admit should succeed")
        .0;
    assert!(admitted.admitted, "{admitted:?}");
    assert!(admitted.signals.len() >= 2);
    assert!((0.0..=1.0).contains(&admitted.agreement));
    assert!(admitted.chance_corrected_agreement.is_some());
    assert!((0.0..=1.0).contains(&admitted.confidence));

    let rejected = server
        .memory_qualify(Parameters(QualifyRequest {
            candidate: "the team chose Rust for the core crates".to_string(),
            goal: None,
        }))
        .await
        .expect("qualify reject should succeed")
        .0;
    assert!(!rejected.admitted, "{rejected:?}");
    assert!(rejected.reason.contains("redundant"));
    assert!(rejected
        .signals
        .iter()
        .any(|signal| signal.name == "novelty" && !signal.passed));
}

#[tokio::test]
async fn memory_anchor_tools_round_trip_with_files_backend() {
    let tempdir = TempDir::new("mcp-anchor");
    let server = MemoryServer::new(tempdir.path());

    server
        .memory_anchor_set(Parameters(AnchorSetRequest {
            current_task: "implement anchor tools".to_string(),
            plan_pointer: Some("docs/self-repair.md#anchor".to_string()),
            last_decisions: Some(vec!["append-only log".to_string()]),
            next_step: "verify MCP round trip".to_string(),
        }))
        .await
        .expect("anchor set should succeed");
    let response = server
        .memory_anchor_get()
        .await
        .expect("anchor get should succeed")
        .0;

    let anchor = response.anchor.expect("anchor should exist");
    assert_eq!(anchor.current_task, "implement anchor tools");
    assert_eq!(anchor.next_step, "verify MCP round trip");
    assert_eq!(anchor.last_decisions, vec!["append-only log"]);
}

#[tokio::test]
async fn memory_session_checkpoint_and_resume_are_cross_agent() {
    let tempdir = TempDir::new("mcp-session");
    let server = MemoryServer::new(tempdir.path());

    server
        .memory_store(Parameters(StoreRequest {
            content: "session scoped implementation detail".to_string(),
            tags: None,
            node_id: Some("node:session-detail".to_string()),
            relations: None,
            source: None,
            confidence: None,
            scope: Some(artesian_mcp::ScopeRequest::Session),
            agent_id: Some("codex".to_string()),
            session_id: Some("session-a".to_string()),
            task_id: Some("task-a".to_string()),
            user_id: Some("user-a".to_string()),
            project: None,
        }))
        .await
        .expect("session memory should store");

    let checkpoint = server
        .memory_session_checkpoint(Parameters(SessionCheckpointRequest {
            agent_id: "codex".to_string(),
            session_id: Some("session-a".to_string()),
            user_id: Some("user-a".to_string()),
            task_id: Some("task-a".to_string()),
            current_task: Some("continue item 7".to_string()),
            next_step: Some("run handoff tests".to_string()),
            plan_pointer: None,
            last_decisions: Some(vec!["agent_id is producer metadata".to_string()]),
            goal: Some("implementation detail".to_string()),
            last_failed_check: Some("clippy failed before handoff".to_string()),
            limit: Some(5),
        }))
        .await
        .expect("checkpoint should succeed")
        .0;

    assert_eq!(checkpoint.summary.handed_off_from.as_deref(), Some("codex"));
    assert_eq!(checkpoint.packet["session"]["handed_off_from"], "codex");

    let resumed = server
        .memory_session_resume(Parameters(SessionResumeRequest {
            session_id: Some("session-a".to_string()),
            user_id: Some("user-a".to_string()),
            task_id: Some("task-a".to_string()),
        }))
        .await
        .expect("resume should succeed")
        .0;
    let state = resumed.packet["restored_working_state"]
        .as_str()
        .expect("state should be text");
    assert!(state.contains("continue item 7"), "{state}");
    assert!(
        state.contains("session scoped implementation detail"),
        "{state}"
    );
    assert_eq!(
        resumed.packet["last_failed_check"],
        "clippy failed before handoff"
    );
}

#[tokio::test]
async fn memory_session_resume_does_not_cross_read_other_users() {
    let tempdir = TempDir::new("mcp-session-isolation");
    let server = MemoryServer::new(tempdir.path());

    for (user, content) in [
        ("user-a", "state that belongs to user a"),
        ("user-b", "state that belongs to user b"),
    ] {
        server
            .memory_session_checkpoint(Parameters(SessionCheckpointRequest {
                agent_id: "codex".to_string(),
                session_id: Some("same-session".to_string()),
                user_id: Some(user.to_string()),
                task_id: Some("same-task".to_string()),
                current_task: Some(content.to_string()),
                next_step: Some("continue".to_string()),
                plan_pointer: None,
                last_decisions: None,
                goal: None,
                last_failed_check: None,
                limit: Some(5),
            }))
            .await
            .expect("checkpoint should succeed");
    }

    let resumed = server
        .memory_session_resume(Parameters(SessionResumeRequest {
            session_id: Some("same-session".to_string()),
            user_id: Some("user-a".to_string()),
            task_id: Some("same-task".to_string()),
        }))
        .await
        .expect("resume should succeed")
        .0;
    let state = resumed.packet["restored_working_state"]
        .as_str()
        .expect("state should be text");
    assert!(state.contains("state that belongs to user a"), "{state}");
    assert!(!state.contains("state that belongs to user b"), "{state}");
}

#[tokio::test]
async fn memory_session_default_key_round_trips_when_identity_is_unset() {
    let tempdir = TempDir::new("mcp-session-default");
    let server = MemoryServer::new(tempdir.path());

    server
        .memory_session_checkpoint(Parameters(SessionCheckpointRequest {
            agent_id: "codex".to_string(),
            session_id: None,
            user_id: None,
            task_id: None,
            current_task: Some("default session task".to_string()),
            next_step: Some("resume default session".to_string()),
            plan_pointer: None,
            last_decisions: None,
            goal: None,
            last_failed_check: None,
            limit: Some(5),
        }))
        .await
        .expect("default checkpoint should succeed");

    let resumed = server
        .memory_session_resume(Parameters(SessionResumeRequest {
            session_id: None,
            user_id: None,
            task_id: None,
        }))
        .await
        .expect("default resume should succeed")
        .0;
    assert!(resumed.packet["restored_working_state"]
        .as_str()
        .expect("state should be text")
        .contains("default session task"));
}

#[tokio::test]
async fn tools_find_is_opt_in_and_reports_token_delta() {
    let tempdir = TempDir::new("mcp-tools-find");
    let disabled = MemoryServer::new(tempdir.path());
    assert!(
        disabled
            .tools_find(Parameters(ToolsFindRequest {
                task: "resume from anchor and search memory".to_string(),
                limit: Some(2),
            }))
            .await
            .is_err(),
        "router should be disabled by default"
    );

    let enabled = MemoryServer::new(tempdir.path()).with_router_enabled(true);
    let response = enabled
        .tools_find(Parameters(ToolsFindRequest {
            task: "resume from anchor and search memory".to_string(),
            limit: Some(2),
        }))
        .await
        .expect("tools.find should run when enabled")
        .0;

    assert!(!response.tools.is_empty());
    assert!(response.prompt_tokens_delta > 0);
    assert!(response
        .tools
        .iter()
        .any(|tool| tool.name == "memory.anchor.get" || tool.name == "memory.find"));
}

#[tokio::test]
async fn orchestration_tools_are_mode_gated_and_agents_list_reflects_catalog() {
    let tempdir = TempDir::new("mcp-orchestration-tools");
    let memory = MemoryServer::new(tempdir.path());
    let memory_tools = memory.visible_tool_names();
    assert!(memory_tools.contains(&"memory.find".to_string()));
    assert!(!memory_tools.contains(&"agents.list".to_string()));
    assert!(!memory_tools.contains(&"orchestrate.delegate".to_string()));
    assert!(!memory_tools.contains(&"team.create".to_string()));
    assert!(!memory_tools.contains(&"team.task.await".to_string()));

    let catalog = AgentCatalog {
        generated_at: Some("test".to_string()),
        agents: vec![AgentCatalogEntry {
            agent: "codex".to_string(),
            command: Some("sh".to_string()),
            reachable: true,
            unreachable_reason: None,
            last_checked: Some("test".to_string()),
            models: vec![AgentModel {
                id: "gpt-5.5".to_string(),
                reachable: true,
                source: "test".to_string(),
            }],
        }],
        roles: Vec::new(),
    };
    let orchestrate = MemoryServer::new(tempdir.path())
        .with_mode(Mode::Orchestrate)
        .with_catalog(catalog.clone());
    let tools = orchestrate.visible_tool_names();
    assert!(tools.contains(&"agents.list".to_string()));
    assert!(tools.contains(&"orchestrate.delegate".to_string()));
    assert!(tools.contains(&"team.create".to_string()));
    assert!(tools.contains(&"team.task.await".to_string()));
    assert!(tools.contains(&"team.cleanup".to_string()));
    let response = orchestrate
        .agents_list()
        .await
        .expect("agents.list should run")
        .0;
    assert_eq!(response.catalog, catalog);
}

#[tokio::test]
async fn agents_list_surfaces_role_definitions() {
    let tempdir = TempDir::new("mcp-team-definitions");
    let definitions = tempdir.join(".agent").join("agents");
    fs::create_dir_all(&definitions).expect("definition dir should exist");
    fs::write(
        definitions.join("worker.md"),
        "---\nname: security-reviewer\nkind: worker\ndescription: Reviews security-sensitive changes.\nagent: sh\n---\nSecurity prompt.\n",
    )
    .expect("definition should write");
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_repo_root(tempdir.path())
        .with_bindings(vec![AgentBinding {
            role: Role::Worker,
            agent: "sh".to_string(),
            model: None,
            command: Some("sh".to_string()),
            args: Vec::new(),
            timeout_seconds: Some(5),
        }]);

    let response = server
        .agents_list()
        .await
        .expect("agents.list should run")
        .0;

    assert!(response
        .catalog
        .roles
        .iter()
        .any(|role| role.name == "security-reviewer" && role.kind == Role::Worker));
}

#[tokio::test]
async fn orchestrate_bind_rejects_unavailable_model() {
    let tempdir = TempDir::new("mcp-bind-unavailable");
    let server = MemoryServer::new(tempdir.path()).with_mode(Mode::Orchestrate);

    let result = server
        .orchestrate_bind(Parameters(BindRequest {
            role: "worker".to_string(),
            agent: "codex".to_string(),
            model: "not-a-codex-model".to_string(),
            command: Some("sh".to_string()),
            args: None,
            timeout_seconds: None,
        }))
        .await;
    let Err(error) = result else {
        panic!("unknown model should fail early");
    };

    assert!(error.to_string().contains("not-a-codex-model"));
}

#[tokio::test]
async fn orchestrate_bind_uses_cached_catalog_models() {
    let tempdir = TempDir::new("mcp-bind-catalog");
    let catalog = AgentCatalog {
        generated_at: Some("test".to_string()),
        agents: vec![AgentCatalogEntry {
            agent: "custom-agent".to_string(),
            command: Some("sh".to_string()),
            reachable: true,
            unreachable_reason: None,
            last_checked: Some("test".to_string()),
            models: vec![AgentModel {
                id: "provider-only-model".to_string(),
                reachable: true,
                source: "provider-api".to_string(),
            }],
        }],
        roles: Vec::new(),
    };
    let server = MemoryServer::new(tempdir.path())
        .with_mode(Mode::Orchestrate)
        .with_catalog(catalog);

    let response = server
        .orchestrate_bind(Parameters(BindRequest {
            role: "worker".to_string(),
            agent: "custom-agent".to_string(),
            model: "provider-only-model".to_string(),
            command: Some("sh".to_string()),
            args: None,
            timeout_seconds: None,
        }))
        .await
        .expect("cached catalog model should bind")
        .0;

    assert_eq!(response.binding.agent, "custom-agent");
    assert_eq!(
        response.binding.model.as_deref(),
        Some("provider-only-model")
    );
}

#[tokio::test]
#[cfg(unix)]
async fn orchestrate_delegate_timeout_uses_supervised_cleanup() {
    let tempdir = TempDir::new("mcp-delegate-timeout");
    let parent_pid_file = tempdir.join("worker.pid");
    let child_pid_file = tempdir.join("grandchild.pid");
    let script = format!(
        "echo $$ > \"{}\"; sleep 30 & echo $! > \"{}\"; wait",
        parent_pid_file.display(),
        child_pid_file.display()
    );
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(vec![AgentBinding {
            role: Role::Worker,
            agent: "codex".to_string(),
            model: Some("gpt-5.5".to_string()),
            command: Some("sh".to_string()),
            args: vec!["-c".to_string(), script],
            timeout_seconds: Some(1),
        }]);

    let result = server
        .orchestrate_delegate_inner(
            DelegateRequest {
                role: "worker".to_string(),
                task: "Keep a child process alive until timeout".to_string(),
                max_output_chars: None,
            },
            CancellationToken::new(),
        )
        .await;
    let Err(error) = result else {
        panic!("delegation should time out");
    };

    assert!(error.to_string().contains("timed out"));
    assert_pid_gone(read_pid(&parent_pid_file));
    assert_pid_gone(read_pid(&child_pid_file));
}

#[tokio::test]
#[cfg(unix)]
async fn orchestrate_delegate_error_redacts_process_secrets() {
    let tempdir = TempDir::new("mcp-delegate-secret-error");
    let secret = "sk-mcp-delegation-secret-123456";
    let script = format!("printf 'api_key={secret}\\n' 1>&2; exit 9");
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(vec![AgentBinding {
            role: Role::Worker,
            agent: "codex".to_string(),
            model: Some("gpt-5.5".to_string()),
            command: Some("sh".to_string()),
            args: vec!["-c".to_string(), script],
            timeout_seconds: Some(5),
        }]);

    let result = server
        .orchestrate_delegate_inner(
            DelegateRequest {
                role: "worker".to_string(),
                task: "Fail with a secret-bearing stderr".to_string(),
                max_output_chars: None,
            },
            CancellationToken::new(),
        )
        .await;
    let Err(error) = result else {
        panic!("delegation should fail");
    };
    let text = error.to_string();

    assert!(!text.contains(secret));
    assert!(text.contains("[REDACTED]"));
}

/// Deterministic regression for the OS-timing flake: a worker that exits without reading
/// its prompt closes the stdin read end, so writing the prompt hits a broken pipe. A prompt
/// larger than the OS pipe buffer forces that path on every platform. Delegation must still
/// capture and redact the worker's stderr rather than aborting with a bare pipe error.
#[tokio::test]
#[cfg(unix)]
async fn orchestrate_delegate_tolerates_unread_stdin_and_redacts() {
    let tempdir = TempDir::new("mcp-delegate-broken-stdin");
    let secret = "sk-mcp-unread-stdin-secret-7890";
    let script = format!("printf 'api_key={secret}\\n' 1>&2; exit 9");
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(vec![AgentBinding {
            role: Role::Worker,
            agent: "codex".to_string(),
            model: Some("gpt-5.5".to_string()),
            command: Some("sh".to_string()),
            args: vec!["-c".to_string(), script],
            timeout_seconds: Some(5),
        }]);

    // Larger than any OS pipe buffer (64 KiB on Linux), so the prompt write blocks and then
    // breaks when the worker exits unread — exercising the broken-pipe path deterministically.
    let big_task = "x".repeat(256 * 1024);
    let result = server
        .orchestrate_delegate_inner(
            DelegateRequest {
                role: "worker".to_string(),
                task: big_task,
                max_output_chars: None,
            },
            CancellationToken::new(),
        )
        .await;
    let Err(error) = result else {
        panic!("delegation should fail");
    };
    let text = error.to_string();

    assert!(!text.contains(secret), "raw secret must not leak: {text}");
    assert!(
        text.contains("[REDACTED]"),
        "worker stderr must be captured and redacted, got: {text}"
    );
}

#[tokio::test]
async fn orchestrate_delegate_truncates_success_output_to_cap() {
    let tempdir = TempDir::new("mcp-delegate-truncate");
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(vec![AgentBinding {
            role: Role::Worker,
            agent: "codex".to_string(),
            model: Some("gpt-5.5".to_string()),
            command: Some("sh".to_string()),
            args: vec![
                "-c".to_string(),
                "printf 'abcdefghijklmnopqrstuvwxyz\\n'".to_string(),
            ],
            timeout_seconds: Some(5),
        }]);

    let response = server
        .orchestrate_delegate_inner(
            DelegateRequest {
                role: "worker".to_string(),
                task: "Return long output".to_string(),
                max_output_chars: Some(10),
            },
            CancellationToken::new(),
        )
        .await
        .expect("delegation should succeed")
        .0;

    assert_eq!(response.status, "done");
    assert_eq!(response.result.as_deref(), Some("abcdefghij"));
    assert!(response.result_truncated);
}

#[tokio::test]
async fn team_lifecycle_uses_definitions_plan_gate_and_cleanup() {
    let tempdir = TempDir::new("mcp-team-lifecycle");
    write_team_definitions(tempdir.path());
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(team_bindings(
            "printf 'worker-result\\n'",
            "printf 'judge-ok\\n'",
        ));

    server
        .team_create(Parameters(TeamCreateRequest {
            id: Some("team".to_string()),
            name: "Test Team".to_string(),
            max_teammates: Some(3),
            plan_approval_required: Some(true),
            plan_approval_roles: None,
        }))
        .await
        .expect("team should create");
    server
        .team_spawn(Parameters(TeamSpawnRequest {
            team_id: "team".to_string(),
            definition: "worker-a".to_string(),
        }))
        .await
        .expect("worker should spawn");
    server
        .team_spawn(Parameters(TeamSpawnRequest {
            team_id: "team".to_string(),
            definition: "judge-a".to_string(),
        }))
        .await
        .expect("judge should spawn");
    let task = server
        .team_task_add(Parameters(TeamTaskAddRequest {
            team_id: "team".to_string(),
            title: "Run team task".to_string(),
            description: Some("Exercise Flume lifecycle".to_string()),
            definition: Some("worker-a".to_string()),
            blockers: None,
        }))
        .await
        .expect("task should add")
        .0;

    let blocked_claim = server
        .team_task_claim(Parameters(TeamTaskClaimRequest {
            team_id: "team".to_string(),
            task_id: Some(task.task_id.clone()),
            teammate: "worker-a".to_string(),
        }))
        .await;
    assert!(
        blocked_claim.is_err(),
        "plan gate should block before review"
    );

    server
        .team_message_inner(
            TeamMessageRequest {
                team_id: "team".to_string(),
                from: "judge-a".to_string(),
                to: Some("worker-a".to_string()),
                kind: TeamMessageKindRequest::Review,
                content: "Plan approved".to_string(),
                task_id: Some(task.task_id.clone()),
                approved: Some(true),
                execute: Some(false),
                resume_packet: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("review should approve plan");
    let claimed = server
        .team_task_claim(Parameters(TeamTaskClaimRequest {
            team_id: "team".to_string(),
            task_id: Some(task.task_id.clone()),
            teammate: "worker-a".to_string(),
        }))
        .await
        .expect("approved task should claim")
        .0;
    assert!(claimed.task.is_some());
    let message = server
        .team_message_inner(
            TeamMessageRequest {
                team_id: "team".to_string(),
                from: "worker-a".to_string(),
                to: Some("worker-a".to_string()),
                kind: TeamMessageKindRequest::Ask,
                content: "Execute the task".to_string(),
                task_id: Some(task.task_id.clone()),
                approved: None,
                execute: Some(true),
                resume_packet: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("worker message should execute")
        .0;
    assert_eq!(
        message.response.as_deref().map(str::trim),
        Some("worker-result")
    );
    let completed = server
        .team_task_complete(Parameters(TeamTaskCompleteRequest {
            team_id: "team".to_string(),
            task_id: task.task_id,
            reviewer: "judge-a".to_string(),
            approved: true,
        }))
        .await
        .expect("judge should complete")
        .0;
    assert_eq!(completed.status, "done");
    let cleaned = server
        .team_cleanup(Parameters(TeamStatusRequest {
            team_id: "team".to_string(),
        }))
        .await
        .expect("cleanup should run")
        .0;
    assert_eq!(cleaned.team["status"], "cleaned-up");
    assert!(
        artesian_process_agent::ProcessSupervisor::new(tempdir.join("spawns"))
            .entries()
            .expect("registry should read")
            .is_empty(),
        "team cleanup should leave no tracked process groups"
    );
}

#[tokio::test]
async fn team_task_await_returns_completed_task_and_progress() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let tempdir = TempDir::new("mcp-team-await-complete");
    let server = await_test_server(&tempdir);
    create_await_test_team(&server).await;
    let task_id = add_await_test_task(&server, "Await completion", "completed by test").await;

    let completer = server.clone();
    let completed_task_id = task_id.clone();
    let complete_handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(120)).await;
        completer
            .team_task_complete(Parameters(TeamTaskCompleteRequest {
                team_id: "team".to_string(),
                task_id: completed_task_id,
                reviewer: "master".to_string(),
                approved: true,
            }))
            .await
            .expect("task should complete");
    });

    let event_count = Arc::new(AtomicU32::new(0));
    let ec = Arc::clone(&event_count);
    let on_progress: Option<TestProgressCallback> = Some(Arc::new(
        move |_p: f64, _t: Option<f64>, _m: Option<String>| {
            ec.fetch_add(1, Ordering::Relaxed);
        },
    ));

    let response = server
        .team_task_await_inner(
            TeamTaskAwaitRequest {
                team_id: "team".to_string(),
                task_id: Some(task_id.clone()),
                task_ids: Vec::new(),
                timeout_secs: Some(5),
                poll_interval_ms: Some(50),
                max_output_chars: None,
            },
            CancellationToken::new(),
            on_progress,
        )
        .await
        .expect("await should complete")
        .0;

    complete_handle.await.expect("completer task should join");
    assert_eq!(response.outcome, "completed");
    assert_eq!(response.task_ids, vec![task_id]);
    assert_eq!(response.tasks.len(), 1);
    assert_eq!(response.tasks[0]["status"], "done");
    assert_eq!(response.tasks[0]["description"], "completed by test");
    assert!(
        event_count.load(Ordering::Relaxed) >= 1,
        "progress callback should fire while waiting"
    );
}

#[tokio::test]
async fn team_task_await_cancels_promptly() {
    let tempdir = TempDir::new("mcp-team-await-cancel");
    let server = await_test_server(&tempdir);
    create_await_test_team(&server).await;
    let task_id = add_await_test_task(&server, "Await cancellation", "left pending").await;
    let ct = CancellationToken::new();
    let cancel = ct.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel.cancel();
    });

    let started = Instant::now();
    let result = server
        .team_task_await_inner(
            TeamTaskAwaitRequest {
                team_id: "team".to_string(),
                task_id: Some(task_id),
                task_ids: Vec::new(),
                timeout_secs: Some(30),
                poll_interval_ms: Some(50),
                max_output_chars: None,
            },
            ct,
            None,
        )
        .await;

    assert!(result.is_err(), "cancelled wait should return an MCP error");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "cancelled wait should not linger"
    );
}

#[tokio::test]
async fn team_task_await_timeout_returns_current_task_state() {
    let tempdir = TempDir::new("mcp-team-await-timeout");
    let server = await_test_server(&tempdir);
    create_await_test_team(&server).await;
    let task_id = add_await_test_task(&server, "Await timeout", "not complete").await;

    let response = server
        .team_task_await_inner(
            TeamTaskAwaitRequest {
                team_id: "team".to_string(),
                task_id: Some(task_id),
                task_ids: Vec::new(),
                timeout_secs: Some(0),
                poll_interval_ms: Some(50),
                max_output_chars: None,
            },
            CancellationToken::new(),
            None,
        )
        .await
        .expect("timeout should be a normal await response")
        .0;

    assert_eq!(response.outcome, "timeout");
    assert_eq!(response.tasks.len(), 1);
    assert_eq!(response.tasks[0]["status"], "todo");
    assert_eq!(response.tasks[0]["description"], "not complete");
}

#[tokio::test]
async fn team_task_await_truncates_long_returned_task_description() {
    let tempdir = TempDir::new("mcp-team-await-truncate");
    let server = await_test_server(&tempdir);
    create_await_test_team(&server).await;
    let long_description = format!("{} tail", "x".repeat(80));
    let task_id = add_await_test_task(&server, "Await truncate", &long_description).await;
    server
        .team_task_complete(Parameters(TeamTaskCompleteRequest {
            team_id: "team".to_string(),
            task_id: task_id.clone(),
            reviewer: "master".to_string(),
            approved: true,
        }))
        .await
        .expect("task should complete");

    let response = server
        .team_task_await_inner(
            TeamTaskAwaitRequest {
                team_id: "team".to_string(),
                task_id: Some(task_id),
                task_ids: Vec::new(),
                timeout_secs: Some(5),
                poll_interval_ms: Some(50),
                max_output_chars: Some(12),
            },
            CancellationToken::new(),
            None,
        )
        .await
        .expect("await should complete")
        .0;

    assert_eq!(response.outcome, "completed");
    assert!(response.tasks_truncated);
    assert_eq!(
        response.tasks[0]["description"]
            .as_str()
            .expect("description should be a string")
            .chars()
            .count(),
        12
    );
    assert_eq!(response.tasks[0]["description_truncated"], true);
}

#[tokio::test]
async fn team_run_blocks_returns_all_results_and_emits_progress() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let tempdir = TempDir::new("mcp-team-run");
    write_team_definitions(tempdir.path());
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(team_bindings(
            "sleep 0.15; printf 'worker-result\\n'",
            "printf 'judge-ok\\n'",
        ));

    let event_count = Arc::new(AtomicU32::new(0));
    let ec = Arc::clone(&event_count);
    let on_progress: Option<TestProgressCallback> = Some(Arc::new(
        move |_p: f64, _t: Option<f64>, _m: Option<String>| {
            ec.fetch_add(1, Ordering::Relaxed);
        },
    ));

    let started = Instant::now();
    let response = server
        .team_run_inner(
            TeamRunRequest {
                team_id: Some("team".to_string()),
                team_name: Some("Atomic Team".to_string()),
                tasks: vec![
                    TeamRunTaskRequest {
                        instruction: "First worker task".to_string(),
                        title: None,
                        role: Some("worker-a".to_string()),
                        agent: None,
                    },
                    TeamRunTaskRequest {
                        instruction: "Second worker task".to_string(),
                        title: None,
                        role: Some("worker-a".to_string()),
                        agent: None,
                    },
                ],
                timeout_secs: Some(5),
                poll_interval_ms: Some(50),
                max_output_chars: Some(100),
            },
            CancellationToken::new(),
            on_progress,
        )
        .await
        .expect("team.run should complete")
        .0;

    assert!(
        started.elapsed() >= Duration::from_millis(120),
        "team.run should block until workers finish"
    );
    assert_eq!(response.outcome, "completed");
    assert_eq!(response.await_outcome, "completed");
    assert_eq!(response.results.len(), 2);
    assert!(response
        .results
        .iter()
        .all(|result| result.status == "done"));
    assert!(response.results.iter().all(|result| {
        result
            .output
            .as_deref()
            .is_some_and(|output| output == "worker-result")
    }));
    assert!(
        event_count.load(Ordering::Relaxed) >= 1,
        "progress callback should fire while team.run waits"
    );
}

#[tokio::test]
async fn team_run_truncates_long_worker_output_to_cap() {
    let tempdir = TempDir::new("mcp-team-run-truncate");
    write_team_definitions(tempdir.path());
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(team_bindings(
            "printf 'abcdefghijklmnopqrstuvwxyz\\n'",
            "printf 'judge-ok\\n'",
        ));

    let response = server
        .team_run_inner(
            TeamRunRequest {
                team_id: Some("team".to_string()),
                team_name: None,
                tasks: vec![TeamRunTaskRequest {
                    instruction: "Return long output".to_string(),
                    title: None,
                    role: Some("worker-a".to_string()),
                    agent: None,
                }],
                timeout_secs: Some(5),
                poll_interval_ms: Some(50),
                max_output_chars: Some(8),
            },
            CancellationToken::new(),
            None,
        )
        .await
        .expect("team.run should complete")
        .0;

    assert_eq!(response.outcome, "completed");
    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].output.as_deref(), Some("abcdefgh"));
    assert!(response.results[0].output_truncated);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg(unix)]
async fn team_run_cancellation_aborts_promptly_and_reaps_workers() {
    let tempdir = TempDir::new("mcp-team-run-cancel");
    write_team_definitions(tempdir.path());
    let parent_pid_file = tempdir.join("worker.pid");
    let child_pid_file = tempdir.join("grandchild.pid");
    let script = format!(
        "echo $$ > \"{}\"; sleep 30 & echo $! > \"{}\"; wait",
        parent_pid_file.display(),
        child_pid_file.display()
    );
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(team_bindings(&script, "printf 'judge-ok\\n'"));
    let ct = CancellationToken::new();
    let cancel = ct.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        cancel.cancel();
    });

    let started = Instant::now();
    let result = server
        .team_run_inner(
            TeamRunRequest {
                team_id: Some("team".to_string()),
                team_name: None,
                tasks: vec![TeamRunTaskRequest {
                    instruction: "Run until cancelled".to_string(),
                    title: None,
                    role: Some("worker-a".to_string()),
                    agent: None,
                }],
                timeout_secs: Some(30),
                poll_interval_ms: Some(50),
                max_output_chars: Some(100),
            },
            ct,
            None,
        )
        .await;

    assert!(result.is_err(), "cancelled team.run should return an error");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "team.run cancellation should not wait for worker timeout"
    );
    assert_pid_gone(read_pid(&parent_pid_file));
    assert_pid_gone(read_pid(&child_pid_file));
}

#[tokio::test]
async fn team_status_remains_non_blocking_snapshot() {
    let tempdir = TempDir::new("mcp-team-status-nonblocking");
    let server = await_test_server(&tempdir);
    create_await_test_team(&server).await;
    let _task_id = add_await_test_task(&server, "Pending task", "status should not wait").await;

    let response = tokio::time::timeout(
        Duration::from_millis(100),
        server.team_status(Parameters(TeamStatusRequest {
            team_id: "team".to_string(),
        })),
    )
    .await
    .expect("team.status should return without waiting")
    .expect("team.status should succeed")
    .0;

    assert_eq!(response.team["id"], "team");
}

#[tokio::test]
async fn team_message_redacts_success_output_and_event_log() {
    let tempdir = TempDir::new("mcp-team-secret");
    write_team_definitions(tempdir.path());
    let secret = "sk-team-mcp-secret-123456";
    let server = MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(team_bindings(
            &format!("printf 'api_key={secret}\\n'"),
            "printf 'judge-ok\\n'",
        ));
    server
        .team_create(Parameters(TeamCreateRequest {
            id: Some("team".to_string()),
            name: "Test Team".to_string(),
            max_teammates: None,
            plan_approval_required: Some(false),
            plan_approval_roles: None,
        }))
        .await
        .expect("team should create");
    server
        .team_spawn(Parameters(TeamSpawnRequest {
            team_id: "team".to_string(),
            definition: "worker-a".to_string(),
        }))
        .await
        .expect("worker should spawn");

    let response = server
        .team_message_inner(
            TeamMessageRequest {
                team_id: "team".to_string(),
                from: "worker-a".to_string(),
                to: Some("worker-a".to_string()),
                kind: TeamMessageKindRequest::Ask,
                content: format!("use token={secret}"),
                task_id: None,
                approved: None,
                execute: Some(true),
                resume_packet: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("message should execute")
        .0;
    let status = server
        .team_status(Parameters(TeamStatusRequest {
            team_id: "team".to_string(),
        }))
        .await
        .expect("status should run")
        .0;
    let text = serde_json::to_string(&serde_json::json!({
        "response": response,
        "status": status,
    }))
    .expect("json should encode");

    assert!(!text.contains(secret));
    assert!(text.contains("[REDACTED]"));
}

#[tokio::test]
async fn memory_tools_store_and_find_with_sqlite_vec_backend() {
    let store = SqliteVecVectorStore::in_memory().expect("sqlite-vec should open");
    let backend = VectorMemoryBackend::with_embedder(
        store,
        VectorMemoryConfig {
            collection: "mcp_sqlite".to_string(),
            dimensions: TEST_DIMENSIONS,
            ..VectorMemoryConfig::new("mcp_sqlite")
        },
        Arc::new(TestEmbedder),
    )
    .expect("backend should construct");
    let server = MemoryServer::with_backend(Arc::new(backend));

    server
        .memory_store(Parameters(StoreRequest {
            content: "MCP sqlite vector memory round trip".to_string(),
            tags: Some(vec!["mcp".to_string()]),
            node_id: Some("node:mcp-sqlite".to_string()),
            relations: None,
            source: None,
            confidence: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("store should succeed");

    let found = server
        .memory_find(Parameters(FindRequest {
            query: "vector".to_string(),
            limit: Some(5),
            node_id: Some("node:mcp-sqlite".to_string()),
            expand: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("find should succeed")
        .0;

    assert_eq!(found.hits.len(), 1);
    assert_eq!(found.hits[0].node_id, "node:mcp-sqlite");
}

fn await_test_server(tempdir: &TempDir) -> MemoryServer {
    MemoryServer::new(tempdir.join("memory"))
        .with_mode(Mode::Orchestrate)
        .with_task_root(tempdir.join("tasks"))
        .with_repo_root(tempdir.path())
        .with_process_registry_dir(tempdir.join("spawns"))
        .with_bindings(Vec::new())
}

async fn create_await_test_team(server: &MemoryServer) {
    server
        .team_create(Parameters(TeamCreateRequest {
            id: Some("team".to_string()),
            name: "Await Test Team".to_string(),
            max_teammates: Some(1),
            plan_approval_required: Some(false),
            plan_approval_roles: None,
        }))
        .await
        .expect("team should create");
}

async fn add_await_test_task(server: &MemoryServer, title: &str, description: &str) -> String {
    server
        .team_task_add(Parameters(TeamTaskAddRequest {
            team_id: "team".to_string(),
            title: title.to_string(),
            description: Some(description.to_string()),
            definition: None,
            blockers: None,
        }))
        .await
        .expect("task should add")
        .0
        .task_id
}

fn write_team_definitions(root: &std::path::Path) {
    let definitions = root.join(".agent").join("agents");
    fs::create_dir_all(&definitions).expect("definition dir should exist");
    fs::write(
        definitions.join("worker-a.md"),
        "---\nname: worker-a\nkind: worker\ndescription: Executes bounded team tasks.\nagent: worker-sh\n---\nWorker prompt.\n",
    )
    .expect("worker definition should write");
    fs::write(
        definitions.join("judge-a.md"),
        "---\nname: judge-a\nkind: judge\ndescription: Reviews team task results.\nagent: judge-sh\n---\nJudge prompt.\n",
    )
    .expect("judge definition should write");
}

fn team_bindings(worker_script: &str, judge_script: &str) -> Vec<AgentBinding> {
    vec![
        AgentBinding {
            role: Role::Worker,
            agent: "worker-sh".to_string(),
            model: None,
            command: Some("sh".to_string()),
            args: vec!["-c".to_string(), worker_script.to_string()],
            timeout_seconds: Some(2),
        },
        AgentBinding {
            role: Role::Judge,
            agent: "judge-sh".to_string(),
            model: None,
            command: Some("sh".to_string()),
            args: vec!["-c".to_string(), judge_script.to_string()],
            timeout_seconds: Some(2),
        },
    ]
}

const TEST_DIMENSIONS: usize = 8;

struct TestEmbedder;

impl TextEmbedder for TestEmbedder {
    fn embed_query(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(test_embedding(text))
    }

    fn embed_passage(&self, text: &str) -> MemoryResult<Vec<f32>> {
        Ok(test_embedding(text))
    }
}

fn test_embedding(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; TEST_DIMENSIONS];
    for token in text.split_whitespace() {
        let index = token.bytes().fold(0usize, |hash, byte| {
            hash.wrapping_mul(31).wrapping_add(byte as usize)
        }) % TEST_DIMENSIONS;
        vector[index] += 1.0;
    }
    let magnitude = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for value in &mut vector {
            *value /= magnitude;
        }
    }
    vector
}

#[cfg(unix)]
fn read_pid(path: &std::path::Path) -> u32 {
    wait_for_file(path);
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .trim()
        .parse()
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

#[cfg(unix)]
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

#[cfg(unix)]
fn assert_pid_gone(pid: u32) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("pid {pid} survived delegated timeout cleanup");
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// `orchestrate.loop` is mode-gated (requires orchestrate or full) and, when goal holds
/// immediately, returns outcome="success" with 0 turns without running the worker.
/// Uses `orchestrate_loop_inner` so the test does not need to construct a `Peer<RoleServer>`.
#[tokio::test]
async fn orchestrate_loop_is_mode_gated_and_succeeds_when_goal_holds_immediately() {
    let tempdir = TempDir::new("mcp-loop");

    // Memory mode: tool must be refused.
    let server_mem = MemoryServer::new(tempdir.path()).with_mode(artesian_core::Mode::Memory);
    let result = server_mem
        .orchestrate_loop_inner(
            LoopRequest {
                goal: "true".to_string(),
                worker: None,
                max_turns: None,
                max_wall_secs: None,
                no_learn: Some(true),
                max_remediation_attempts: None,
            },
            CancellationToken::new(),
            None,
        )
        .await;
    assert!(
        result.is_err(),
        "orchestrate.loop must be gated in memory mode"
    );
    let err = result.err().expect("expected error from mode gate");
    assert!(
        err.message.contains("orchestration") || err.message.contains("mode"),
        "error should mention orchestration gate: {}",
        err.message
    );

    // Orchestrate mode: `true` exits 0 immediately — goal already holds, 0 turns.
    let server_orch = MemoryServer::new(tempdir.path()).with_mode(artesian_core::Mode::Orchestrate);
    let response = server_orch
        .orchestrate_loop_inner(
            LoopRequest {
                goal: "true".to_string(),
                worker: None,
                max_turns: Some(5),
                max_wall_secs: None,
                no_learn: Some(true),
                max_remediation_attempts: None,
            },
            CancellationToken::new(),
            None,
        )
        .await
        .expect("orchestrate.loop should succeed when goal holds immediately")
        .0;
    assert_eq!(response.outcome, "success");
    assert_eq!(response.turns, 0);
    assert!(!response.run_log_path.is_empty());
}

/// `orchestrate.loop` runs to max-turns when the goal never holds.
/// Uses `orchestrate_loop_inner` so the test does not need a `Peer<RoleServer>`.
#[tokio::test]
async fn orchestrate_loop_reaches_max_turns_when_goal_never_holds() {
    let tempdir = TempDir::new("mcp-loop-max-turns");
    let server = MemoryServer::new(tempdir.path()).with_mode(artesian_core::Mode::Orchestrate);

    let response = server
        .orchestrate_loop_inner(
            LoopRequest {
                goal: "false".to_string(),
                worker: Some("true".to_string()),
                max_turns: Some(2),
                max_wall_secs: None,
                no_learn: Some(true),
                max_remediation_attempts: Some(0), // disable escalation so max-turns fires
            },
            CancellationToken::new(),
            None,
        )
        .await
        .expect("orchestrate.loop should return a report even on max-turns")
        .0;

    assert_eq!(response.outcome, "max-turns");
    assert_eq!(response.turns, 2);
    assert!(
        response.run_log_path.ends_with(".jsonl"),
        "run_log_path should be a .jsonl file: {}",
        response.run_log_path
    );
}

/// `orchestrate_loop_inner` forwards the `on_progress` callback into `run_loop_core`.
/// This exercises that a client providing a progressToken would receive notifications.
#[tokio::test]
#[allow(clippy::type_complexity)]
async fn orchestrate_loop_inner_emits_progress_events() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let tempdir = TempDir::new("mcp-loop-progress");
    let server = MemoryServer::new(tempdir.path()).with_mode(artesian_core::Mode::Orchestrate);

    let event_count = Arc::new(AtomicU32::new(0));
    let ec = Arc::clone(&event_count);
    // Use the concrete type so we do not need to import the flume crate directly.
    #[allow(clippy::type_complexity)]
    let on_progress: Option<Arc<dyn Fn(f64, Option<f64>, Option<String>) + Send + Sync>> = Some(
        Arc::new(move |_p: f64, _t: Option<f64>, _m: Option<String>| {
            ec.fetch_add(1, Ordering::Relaxed);
        }),
    );

    // `true` exits 0 immediately — goal already held, so 0 turns — but the initial verify
    // may or may not call on_progress depending on implementation. Use `false` + 1 turn
    // to guarantee at least one turn-start event.
    let response = server
        .orchestrate_loop_inner(
            LoopRequest {
                goal: "false".to_string(),
                worker: Some("true".to_string()),
                max_turns: Some(1),
                max_wall_secs: None,
                no_learn: Some(true),
                max_remediation_attempts: Some(0),
            },
            CancellationToken::new(),
            on_progress,
        )
        .await
        .expect("should return a report")
        .0;

    assert_eq!(response.outcome, "max-turns");
    let count = event_count.load(Ordering::Relaxed);
    assert!(
        count >= 1,
        "expected at least 1 progress event (turn-start); got {count}"
    );
}

/// A pre-cancelled token stops the loop before any turn runs.
#[tokio::test]
async fn orchestrate_loop_inner_cancel_stops_loop() {
    let tempdir = TempDir::new("mcp-loop-cancel");
    let server = MemoryServer::new(tempdir.path()).with_mode(artesian_core::Mode::Orchestrate);

    let ct = CancellationToken::new();
    ct.cancel(); // already fired

    let response = server
        .orchestrate_loop_inner(
            LoopRequest {
                goal: "false".to_string(),
                worker: Some("sleep 60".to_string()),
                max_turns: Some(10),
                max_wall_secs: None,
                no_learn: Some(true),
                max_remediation_attempts: Some(0),
            },
            ct,
            None,
        )
        .await
        .expect("inner should return a report even when cancelled")
        .0;

    assert_eq!(
        response.outcome, "cancelled",
        "pre-cancelled token must yield outcome 'cancelled'"
    );
    assert_eq!(
        response.turns, 0,
        "no turns should run when already cancelled"
    );
}

// ── OCF cross-agent session continuity (end-to-end) ──────────────────────────────────────────

/// Full OCF cross-agent flow:
///
/// 1. Codex checkpoints a keyed session (user=u1, session=s1, task="DPT-4477 add dag_id/run_id")
///    with current_task, next_step, last_failed_check, and a session-scoped memory record.
/// 2. Claude resumes by task query "DPT-4477" via `memory.session.resume_by_task`.
/// 3. Assert the packet restores current_task/next_step/last_failed_check and the relevant memory.
/// 4. Assert `handed_off_from == "codex"` (proving the cross-agent identity is preserved).
/// 5. Non-matching query returns an error (clean miss).
/// 6. Exact-id handoff (`memory.session.resume`) still works.
#[tokio::test]
async fn ocf_cross_agent_session_continuity_end_to_end() {
    let tempdir = TempDir::new("mcp-ocf-e2e");
    let server = MemoryServer::new(tempdir.path());

    // Store a session-scoped memory that the resumed agent should see.
    server
        .memory_store(Parameters(StoreRequest {
            content: "dag_id and run_id must be propagated via XCom".to_string(),
            tags: Some(vec!["airflow".to_string(), "design".to_string()]),
            node_id: Some("node:dpt-4477-xcom".to_string()),
            relations: None,
            source: Some("DPT-4477".to_string()),
            confidence: Some(1.0),
            scope: Some(artesian_mcp::ScopeRequest::Session),
            agent_id: Some("codex".to_string()),
            session_id: Some("s1".to_string()),
            task_id: Some("DPT-4477 add dag_id/run_id".to_string()),
            user_id: Some("u1".to_string()),
            project: None,
        }))
        .await
        .expect("session-scoped memory should store");

    // Store a second memory so we can confirm relevance recall.
    server
        .memory_store(Parameters(StoreRequest {
            content: "backfill runs must not re-trigger the downstream sensor".to_string(),
            tags: Some(vec!["invariant".to_string()]),
            node_id: Some("node:dpt-4477-inv".to_string()),
            relations: None,
            source: Some("DPT-4477".to_string()),
            confidence: Some(1.0),
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("invariant memory should store");

    // Codex checkpoints the keyed session.
    let checkpoint = server
        .memory_session_checkpoint(Parameters(SessionCheckpointRequest {
            agent_id: "codex".to_string(),
            user_id: Some("u1".to_string()),
            session_id: Some("s1".to_string()),
            task_id: Some("DPT-4477 add dag_id/run_id".to_string()),
            current_task: Some("DPT-4477 add dag_id/run_id to Airflow operator".to_string()),
            next_step: Some("wire XCom push in execute() and add unit test".to_string()),
            plan_pointer: Some("docs/plan.md#step-3".to_string()),
            last_decisions: Some(vec![
                "use XCom over env vars for testability".to_string(),
                "scope to task-scope memory to avoid session bleed".to_string(),
            ]),
            goal: Some("DPT-4477 dag_id run_id XCom".to_string()),
            last_failed_check: Some(
                "mypy: Argument 1 to XCom.set has incompatible type".to_string(),
            ),
            limit: Some(8),
        }))
        .await
        .expect("codex checkpoint should succeed")
        .0;

    // Verify the checkpoint itself records the producer.
    assert_eq!(
        checkpoint.summary.handed_off_from.as_deref(),
        Some("codex"),
        "checkpoint summary should record codex as producer"
    );
    assert_eq!(
        checkpoint.packet["session"]["handed_off_from"], "codex",
        "checkpoint packet session block should record codex"
    );

    // Claude resumes by task query "DPT-4477" — no session_id needed.
    let resumed = server
        .memory_session_resume_by_task(Parameters(SessionResumeByTaskRequest {
            task_query: "DPT-4477".to_string(),
            user_id: Some("u1".to_string()),
        }))
        .await
        .expect("resume_by_task should match DPT-4477")
        .0;

    // The matched session should be the one Codex checkpointed.
    assert!(
        resumed.matched_task_id.contains("DPT-4477"),
        "matched_task_id should contain DPT-4477, got: {}",
        resumed.matched_task_id
    );

    // Cross-agent identity: handed_off_from == "codex".
    assert_eq!(
        resumed.packet["session"]["handed_off_from"], "codex",
        "resumed packet should show handed_off_from=codex (cross-agent proof)"
    );

    // The working state should contain current_task and next_step.
    let state = resumed.packet["restored_working_state"]
        .as_str()
        .expect("restored_working_state should be a string");
    assert!(
        state.contains("DPT-4477 add dag_id/run_id to Airflow operator"),
        "current_task missing from restored state: {state}"
    );
    assert!(
        state.contains("wire XCom push in execute()"),
        "next_step missing from restored state: {state}"
    );

    // last_failed_check must survive the handoff.
    assert_eq!(
        resumed.packet["last_failed_check"], "mypy: Argument 1 to XCom.set has incompatible type",
        "last_failed_check should survive the handoff"
    );

    // The session-scoped memory (stored by codex) must appear in the restored state.
    assert!(
        state.contains("dag_id and run_id must be propagated via XCom"),
        "session-scoped memory should appear in restored state: {state}"
    );

    // Non-matching query must return an error (clean miss).
    let miss = server
        .memory_session_resume_by_task(Parameters(SessionResumeByTaskRequest {
            task_query: "NONEXISTENT-9999".to_string(),
            user_id: Some("u1".to_string()),
        }))
        .await;
    assert!(
        miss.is_err(),
        "non-matching query should return an error, not a session"
    );

    // Exact-id handoff via the original memory.session.resume must still work.
    let exact_resumed = server
        .memory_session_resume(Parameters(SessionResumeRequest {
            session_id: Some("s1".to_string()),
            user_id: Some("u1".to_string()),
            task_id: Some("DPT-4477 add dag_id/run_id".to_string()),
        }))
        .await
        .expect("exact-id resume should succeed")
        .0;
    assert_eq!(
        exact_resumed.packet["session"]["handed_off_from"], "codex",
        "exact-id resume should also show handed_off_from=codex"
    );
    let exact_state = exact_resumed.packet["restored_working_state"]
        .as_str()
        .expect("exact resume state should be a string");
    assert!(
        exact_state.contains("DPT-4477 add dag_id/run_id to Airflow operator"),
        "exact-id resume should restore current_task: {exact_state}"
    );
}

/// Governed skill-memory: `memory.learn` + `memory.skills`.
///
/// Verifies:
/// - `memory.learn` commits a skill record tagged `skill` with the right title, content,
///   and provenance (`source`).
/// - `memory.skills` lists all skills with title, access_count, source, last_access.
/// - Re-learning the same title+content is idempotent (same node_id, no duplicate).
/// - A learned skill is retrievable via `memory.find` (confirms tag + content indexing).
/// - `memory.skills` with `by_usage=true` orders skills by access_count descending
///   after the fire-and-forget access write settles.
#[tokio::test]
async fn memory_learn_and_skills_list() {
    let tempdir = TempDir::new("mcp-learn");
    let server = MemoryServer::new(tempdir.path());

    // ── learn AlphaSkill ────────────────────────────────────────────────────
    let alpha = server
        .memory_learn(Parameters(LearnRequest {
            title: "AlphaSkill".to_string(),
            content: "Use alpha pattern for optimal throughput".to_string(),
            sources: Some(vec!["docs/alpha.md".to_string()]),
            tags: Some(vec!["performance".to_string()]),
            procedure: None,
        }))
        .await
        .expect("learn AlphaSkill should succeed")
        .0;

    assert!(
        alpha.node_id.starts_with("skill:"),
        "node_id should have skill: prefix; got {}",
        alpha.node_id
    );

    // ── idempotency: same title+content → same node_id, no duplicate ────────
    let alpha_again = server
        .memory_learn(Parameters(LearnRequest {
            title: "AlphaSkill".to_string(),
            content: "Use alpha pattern for optimal throughput".to_string(),
            sources: Some(vec!["docs/alpha.md".to_string()]),
            tags: Some(vec!["performance".to_string()]),
            procedure: None,
        }))
        .await
        .expect("re-learn should succeed")
        .0;

    assert_eq!(
        alpha.node_id, alpha_again.node_id,
        "re-learning identical title+content must return the same node_id"
    );

    // ── learn BetaSkill (no sources → falls back to artesian-learn) ─────────
    let _beta = server
        .memory_learn(Parameters(LearnRequest {
            title: "BetaSkill".to_string(),
            content: "Use beta pattern for low latency".to_string(),
            sources: None,
            tags: None,
            procedure: None,
        }))
        .await
        .expect("learn BetaSkill should succeed")
        .0;

    // ── skills list: both skills appear with expected fields ─────────────────
    let skills = server
        .memory_skills(Parameters(SkillsRequest {
            limit: Some(10),
            by_usage: Some(false),
        }))
        .await
        .expect("memory.skills should succeed")
        .0;

    assert_eq!(
        skills.skills.len(),
        2,
        "should list exactly 2 skills; got {:?}",
        skills
            .skills
            .iter()
            .map(|s| s.node_id.as_str())
            .collect::<Vec<_>>()
    );

    let alpha_hit = skills
        .skills
        .iter()
        .find(|s| s.title.as_deref() == Some("AlphaSkill"))
        .expect("AlphaSkill should appear in skills list");
    assert_eq!(
        alpha_hit.source.as_deref(),
        Some("docs/alpha.md"),
        "AlphaSkill provenance should be docs/alpha.md"
    );
    assert!(
        alpha_hit.content.contains("alpha pattern"),
        "AlphaSkill content should include body text"
    );
    assert!(
        alpha_hit.content.starts_with("# AlphaSkill"),
        "AlphaSkill content should start with title heading"
    );

    let beta_hit = skills
        .skills
        .iter()
        .find(|s| s.title.as_deref() == Some("BetaSkill"))
        .expect("BetaSkill should appear in skills list");
    assert_eq!(
        beta_hit.source.as_deref(),
        Some("artesian-learn"),
        "BetaSkill with no sources should fall back to artesian-learn"
    );

    // ── skill retrieval: AlphaSkill is reachable via memory.find ───────────
    // Pin to AlphaSkill's node_id so the query is exact and BetaSkill is excluded.
    let found = server
        .memory_find(Parameters(FindRequest {
            query: "alpha pattern".to_string(),
            limit: Some(5),
            node_id: Some(alpha.node_id.clone()),
            expand: None,
            scope: None,
            agent_id: None,
            session_id: None,
            task_id: None,
            user_id: None,
            project: None,
        }))
        .await
        .expect("find should succeed")
        .0;

    assert!(
        found.hits.iter().any(|h| h.node_id == alpha.node_id),
        "AlphaSkill should be retrievable via memory.find with its node_id; got hits: {:?}",
        found
            .hits
            .iter()
            .map(|h| h.node_id.as_str())
            .collect::<Vec<_>>()
    );

    // ── skills list: both skills appear with expected fields ─────────────────
    let skills = server
        .memory_skills(Parameters(SkillsRequest {
            limit: Some(10),
            by_usage: Some(false),
        }))
        .await
        .expect("memory.skills should succeed")
        .0;

    assert_eq!(
        skills.skills.len(),
        2,
        "should list exactly 2 skills; got {:?}",
        skills
            .skills
            .iter()
            .map(|s| s.node_id.as_str())
            .collect::<Vec<_>>()
    );

    let alpha_hit = skills
        .skills
        .iter()
        .find(|s| s.title.as_deref() == Some("AlphaSkill"))
        .expect("AlphaSkill should appear in skills list");
    assert_eq!(
        alpha_hit.source.as_deref(),
        Some("docs/alpha.md"),
        "AlphaSkill provenance should be docs/alpha.md"
    );
    assert!(
        alpha_hit.content.contains("alpha pattern"),
        "AlphaSkill content should include body text"
    );
    assert!(
        alpha_hit.content.starts_with("# AlphaSkill"),
        "AlphaSkill content should start with title heading"
    );

    let beta_hit = skills
        .skills
        .iter()
        .find(|s| s.title.as_deref() == Some("BetaSkill"))
        .expect("BetaSkill should appear in skills list");
    assert_eq!(
        beta_hit.source.as_deref(),
        Some("artesian-learn"),
        "BetaSkill with no sources should fall back to artesian-learn"
    );

    // ── memory.skills by_usage=true: flag accepted, both skills returned ─────
    // Fire-and-forget access_count writes from find calls above are ephemeral within
    // an integration test (the spawned tokio tasks may not settle before the next call).
    // The access_count sort contract is tested in aquifer/tests/access_tracking.rs where
    // the backend is driven directly. Here we only verify the flag is wired end-to-end.
    let skills_by_usage = server
        .memory_skills(Parameters(SkillsRequest {
            limit: Some(10),
            by_usage: Some(true),
        }))
        .await
        .expect("memory.skills by_usage should succeed")
        .0;

    assert_eq!(
        skills_by_usage.skills.len(),
        2,
        "by_usage list should contain both skills"
    );
    assert!(
        skills_by_usage
            .skills
            .iter()
            .any(|s| s.title.as_deref() == Some("AlphaSkill")),
        "by_usage list should include AlphaSkill"
    );
    assert!(
        skills_by_usage
            .skills
            .iter()
            .any(|s| s.title.as_deref() == Some("BetaSkill")),
        "by_usage list should include BetaSkill"
    );
    // All returned skills carry the usage field (even when zero).
    for skill in &skills_by_usage.skills {
        let _ = skill.access_count; // field must be present (u32, always serialized)
    }
}

#[tokio::test]
async fn memory_skill_replay_dry_run_and_execute() {
    let tempdir = TempDir::new("mcp-skill-replay");
    let mut config = artesian_core::ArtesianConfig::memory_files(
        tempdir.path().display().to_string(),
        Vec::new(),
    )
    .memory;
    config.track_savings = false;
    let server = MemoryServer::from_config(&config).expect("server should open");

    server
        .memory_learn(Parameters(LearnRequest {
            title: "ReplaySkill".to_string(),
            content: "Replay this guarded procedure".to_string(),
            sources: None,
            tags: None,
            procedure: Some(vec![SkillProcedureStep {
                run: "echo mcp-replayed".to_string(),
                guard: Some("true".to_string()),
            }]),
        }))
        .await
        .expect("learn should succeed");

    let skills = server
        .memory_skills(Parameters(SkillsRequest {
            limit: Some(10),
            by_usage: Some(false),
        }))
        .await
        .expect("skills should succeed")
        .0;
    let replay_skill = skills
        .skills
        .iter()
        .find(|skill| skill.title.as_deref() == Some("ReplaySkill"))
        .expect("ReplaySkill should be listed");
    assert_eq!(
        replay_skill.procedure.as_ref().map(Vec::len),
        Some(1),
        "procedure should be visible in memory.skills"
    );

    let dry_run = server
        .memory_skill_replay(Parameters(SkillReplayRequest {
            title: "ReplaySkill".to_string(),
            execute: None,
        }))
        .await
        .expect("dry-run replay should succeed")
        .0;
    assert_eq!(dry_run.status, "dry-run");
    assert_eq!(dry_run.steps[0].guard_status, "not-run");
    assert_eq!(dry_run.steps[0].run_status, "not-run");

    let executed = server
        .memory_skill_replay(Parameters(SkillReplayRequest {
            title: "ReplaySkill".to_string(),
            execute: Some(true),
        }))
        .await
        .expect("execute replay should succeed")
        .0;
    assert_eq!(executed.status, "success");
    assert!(!executed.fallback);
    assert_eq!(executed.steps[0].guard_status, "passed");
    assert_eq!(executed.steps[0].run_status, "passed");
    assert_eq!(
        executed.steps[0].run_output.as_deref(),
        Some("mcp-replayed")
    );
}
