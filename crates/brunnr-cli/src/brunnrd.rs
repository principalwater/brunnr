// SPDX-License-Identifier: Apache-2.0

use std::{env, path::PathBuf, time::Duration};

use anyhow::{bail, Result};
use brunnr_core::Mode;
use clap::Parser;
use serde_json::json;

mod runtime;
use runtime::{build_orchestrator, load_config, process_supervisor_from_config, shutdown_signal};

const DEFAULT_CONFIG: &str = "brunnr.toml";

#[derive(Debug, Parser)]
#[command(name = "brunnrd", about = "Brunnr orchestration daemon")]
struct Cli {
    #[arg(long, default_value = DEFAULT_CONFIG)]
    config: PathBuf,
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 1000)]
    interval_millis: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli.config)?;
    if !matches!(config.mode, Mode::Orchestrate | Mode::Full) {
        bail!(
            "brunnrd requires mode orchestrate or full, got {:?}",
            config.mode
        );
    }
    let root = cli
        .root
        .unwrap_or_else(|| PathBuf::from(&config.memory.root));
    let repo_root = env::current_dir()?;
    let supervisor = process_supervisor_from_config(&config, &repo_root);
    let reaped = supervisor.reap_stale()?;
    if reaped.terminated > 0 {
        eprintln!(
            "reaped stale process groups before daemon start: terminated={}",
            reaped.terminated
        );
    }
    let mut orchestrator = build_orchestrator(config, root, repo_root, cli.dry_run)?;

    loop {
        let report = tokio::select! {
            report = orchestrator.run_once() => report?,
            signal = shutdown_signal() => {
                let signal = signal?;
                let report = supervisor.terminate_current_owner()?;
                eprintln!(
                    "brunnrd received {signal}; terminated tracked process groups={}",
                    report.terminated
                );
                return Ok(());
            }
        };
        println!(
            "{}",
            serde_json::to_string(&json!({
                "dispatched": report.dispatched,
                "completed": report.completed,
                "blocked": report.blocked,
                "idle": report.idle,
                "events": orchestrator.run_log().events.len()
            }))?
        );
        if cli.once {
            break;
        }
        tokio::select! {
            signal = shutdown_signal() => {
                let signal = signal?;
                let report = supervisor.terminate_current_owner()?;
                eprintln!(
                    "brunnrd received {signal}; terminated tracked process groups={}",
                    report.terminated
                );
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(cli.interval_millis)) => {}
        }
    }
    Ok(())
}
