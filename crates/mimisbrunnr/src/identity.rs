// SPDX-License-Identifier: Apache-2.0

use sha2::{Digest, Sha256};

use crate::{MemoryId, StoreMemory};

pub(crate) fn stable_memory_id(memory: &StoreMemory) -> MemoryId {
    let mut hasher = Sha256::new();
    hasher.update(memory.content.as_bytes());
    hasher.update(format!("{:?}", memory.tier).as_bytes());
    if let Some(node_id) = &memory.node_id {
        hasher.update(node_id.as_bytes());
    }
    MemoryId::new(format!("{:x}", hasher.finalize()))
}
