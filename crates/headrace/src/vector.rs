// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use aquifer::{MemoryBackend, MemoryQuery, MemoryTier, StoreMemory};
use futures_util::{future::BoxFuture, FutureExt};

use crate::{
    ClaimRequest, FilesTaskStore, NewTask, Task, TaskImportOutcome, TaskResult, TaskStore,
    TransitionTask,
};

#[derive(Clone)]
pub struct VectorTaskStore {
    source: FilesTaskStore,
    memory: Arc<dyn MemoryBackend>,
}

impl VectorTaskStore {
    pub fn new(source: FilesTaskStore, memory: Arc<dyn MemoryBackend>) -> Self {
        Self { source, memory }
    }

    pub fn source(&self) -> &FilesTaskStore {
        &self.source
    }

    pub async fn import_task(&self, task: Task) -> TaskResult<TaskImportOutcome> {
        let outcome = self.source.import_task(task).await?;
        self.index_task(outcome.task()).await?;
        Ok(outcome)
    }

    async fn index_task(&self, task: &Task) -> TaskResult<()> {
        self.memory
            .store(StoreMemory {
                content: format!("{} {}", task.title, task.description),
                tags: vec!["task".to_string(), task.status.directory().to_string()],
                metadata: [
                    ("task_id".to_string(), task.id.clone()),
                    (
                        "task_status".to_string(),
                        task.status.directory().to_string(),
                    ),
                ]
                .into_iter()
                .collect(),
                tier: MemoryTier::L1Atom,
                node_id: Some(format!("task:{}", task.id)),
                created_at: Some(task.updated_at),
                scope: None,
                agent_id: None,
                session_id: None,
                task_id: Some(task.id.clone()),
                user_id: None,
                project: None,
                source: None,
                confidence: None,
                relations: Vec::new(),
            })
            .await?;
        Ok(())
    }
}

impl TaskStore for VectorTaskStore {
    fn create(&self, task: NewTask) -> BoxFuture<'_, TaskResult<Task>> {
        async move {
            let task = self.source.create(task).await?;
            self.index_task(&task).await?;
            Ok(task)
        }
        .boxed()
    }

    fn claim(&self, request: ClaimRequest) -> BoxFuture<'_, TaskResult<Option<Task>>> {
        async move {
            let task = self.source.claim(request).await?;
            if let Some(task) = &task {
                self.index_task(task).await?;
            }
            Ok(task)
        }
        .boxed()
    }

    fn transition(&self, transition: TransitionTask) -> BoxFuture<'_, TaskResult<Task>> {
        async move {
            let task = self.source.transition(transition).await?;
            self.index_task(&task).await?;
            Ok(task)
        }
        .boxed()
    }

    fn get(&self, id: &str) -> BoxFuture<'_, TaskResult<Option<Task>>> {
        self.source.get(id)
    }

    fn list(&self) -> BoxFuture<'_, TaskResult<Vec<Task>>> {
        self.source.list()
    }

    fn find(&self, query: &str) -> BoxFuture<'_, TaskResult<Vec<Task>>> {
        let query = query.to_string();
        async move {
            let hits = self
                .memory
                .find(MemoryQuery::new(query.clone()).with_limit(25))
                .await?;
            let mut tasks = Vec::new();
            for hit in hits {
                let Some(task_id) = hit.record.metadata.get("task_id") else {
                    continue;
                };
                if let Some(task) = self.source.get(task_id).await? {
                    if !tasks.iter().any(|existing: &Task| existing.id == task.id) {
                        tasks.push(task);
                    }
                }
            }
            if tasks.is_empty() {
                return self.source.find(&query).await;
            }
            Ok(tasks)
        }
        .boxed()
    }
}
