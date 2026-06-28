// SPDX-License-Identifier: Apache-2.0

use sha2::{Digest, Sha256};

use crate::{MemoryId, StoreMemory};

pub fn stable_memory_id(memory: &StoreMemory) -> MemoryId {
    let mut hasher = Sha256::new();
    hasher.update(memory.content.as_bytes());
    hasher.update(format!("{:?}", memory.tier).as_bytes());
    if let Some(node_id) = &memory.node_id {
        hasher.update(node_id.as_bytes());
    }
    if let Some(scope) = memory.scope {
        hasher.update(scope.as_str().as_bytes());
    }
    for value in [
        &memory.agent_id,
        &memory.session_id,
        &memory.task_id,
        &memory.user_id,
        &memory.project,
    ]
    .into_iter()
    .flatten()
    {
        hasher.update(value.as_bytes());
    }
    MemoryId::new(format!("{:x}", hasher.finalize()))
}
