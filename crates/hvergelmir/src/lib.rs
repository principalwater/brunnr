// SPDX-License-Identifier: Apache-2.0

//! Optional Hvergelmir sandbox runtime seam.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxProfile {
    pub enabled: bool,
    pub image: Option<String>,
    pub allow_network: bool,
    pub mounted_paths: Vec<String>,
}
