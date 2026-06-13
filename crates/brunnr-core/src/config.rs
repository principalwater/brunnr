// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::Role;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Memory,
    Orchestrate,
    Full,
    Advanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryBackendKind {
    Files,
    SqliteVec,
    Qdrant,
    TencentDb,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub backend: MemoryBackendKind,
    pub root: String,
    #[serde(default = "default_memory_collection")]
    pub collection: String,
    #[serde(default)]
    pub qdrant_url: Option<String>,
    #[serde(default)]
    pub qdrant_api_key_env: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentBinding {
    pub role: Role,
    pub agent: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrunnrConfig {
    pub mode: Mode,
    pub memory: MemoryConfig,
    pub agents: Vec<AgentBinding>,
}

impl BrunnrConfig {
    pub fn memory_files(root: impl Into<String>, agents: Vec<AgentBinding>) -> Self {
        Self {
            mode: Mode::Memory,
            memory: MemoryConfig {
                backend: MemoryBackendKind::Files,
                root: root.into(),
                collection: default_memory_collection(),
                qdrant_url: None,
                qdrant_api_key_env: None,
            },
            agents,
        }
    }

    pub fn from_toml(input: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(input)
    }

    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

fn default_memory_collection() -> String {
    "brunnr-memory".to_string()
}
