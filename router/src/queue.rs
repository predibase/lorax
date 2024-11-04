use core::fmt;
use std::{
    backtrace::Backtrace,
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use tokio::{sync::Notify, time::Instant};
use tracing::info_span;

use crate::{adapter::Adapter, batch::Entry};

#[derive(Debug, PartialEq)]
pub(crate) enum AdapterStatus {
    Downloading,
    Downloaded,
    Ready,
    Errored,
}

impl fmt::Display for AdapterStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug)]
pub(crate) struct AdapterEvent {
    /// Adapter readniess task notifier
    pub batching_task: Notify,
}

/// Queue State
#[derive(Debug)]
pub(crate) struct QueueState {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Adapter index
    adapter: Adapter,

    /// Adapter status
    status: AdapterStatus,

    /// Cost as a fraction of the adapter memory budget
    cost: Option<f32>,

    /// Timestamp when the adapter was last activated
    activation_ts: Option<Instant>,

    /// Adapter event
    event: Arc<AdapterEvent>,
}

impl QueueState {
    pub(crate) fn new(adapter: Adapter, event: Arc<AdapterEvent>) -> Self {
        let status = AdapterStatus::Downloading;
        Self {
            entries: VecDeque::with_capacity(128),
            adapter,
            status,
            cost: None,
            activation_ts: None,
            event,
        }
    }

    /// Append an entry to the queue
    pub(crate) fn append(&mut self, entry_id: u64, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push_back((entry_id, entry));
    }

    /// Prepend an entry to the front of the queue
    pub(crate) fn push_front(&mut self, entry_id: u64, entry: Entry) {
        self.entries.push_front((entry_id, entry));
    }

    // Is empty
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // Peeks at the front of the queue and returns timestamp of the oldest entry, if present
    pub(crate) fn peek(&self) -> Option<Instant> {
        self.entries.front().map(|(_, entry)| entry.queue_time)
    }

    // Pops the front of the queue and returns the oldest entry, if present
    pub(crate) fn pop(&mut self) -> Option<(u64, Entry, Option<Instant>)> {
        self.entries
            .pop_front()
            .map(|(id, entry)| (id, entry, self.peek()))
    }

    pub(crate) fn entries(&self) -> &VecDeque<(u64, Entry)> {
        &self.entries
    }

    pub(crate) fn drain(&mut self) -> std::collections::vec_deque::Drain<(u64, Entry)> {
        self.entries.drain(..)
    }

    pub(crate) fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    pub(crate) fn set_status(&mut self, status: AdapterStatus) {
        self.status = status;
        self.event.batching_task.notify_one();
        tracing::info!(
            "set adapter {} status to {}",
            self.adapter.as_string(),
            self.status
        );
    }

    pub(crate) fn status(&self) -> &AdapterStatus {
        &self.status
    }

    pub(crate) fn set_cost(&mut self, cost: f32) {
        self.cost = Some(cost);
    }

    pub(crate) fn cost(&self) -> Option<f32> {
        self.cost
    }

    pub(crate) fn set_activation_ts(&mut self, ts: Instant) {
        self.activation_ts = Some(ts);
    }

    pub(crate) fn activation_ts(&self) -> Option<Instant> {
        self.activation_ts
    }
}

#[derive(Debug)]
pub(crate) struct AdapterQueuesState {
    /// Map of adapter key to queue
    pub queue_map: HashMap<Adapter, QueueState>,

    /// Adapters that are currently not in use, but have entries in queue
    pending_adapters: VecDeque<Adapter>,

    /// Adapters that are currently in use
    active_adapters: VecDeque<Adapter>,

    /// Adapters that are currently being tracked as pending or active
    tracked_adapters: HashSet<Adapter>,

    /// Number of adapters that can be active at a time
    max_active_adapters: usize,

    /// Fraction of adapter memory budget remaining to allocate to new adapters
    memory_budget_remaining: f32,

    /// Maximum time an adapter is allowed to be active before exchanging out
    max_active_time: Duration,

    /// Id of the next entry
    next_id: u64,
}

impl AdapterQueuesState {
    pub(crate) fn new(max_active_adapters: usize, adapter_cycle_time_s: u64) -> Self {
        let queue_map = HashMap::new();
        let pending_adapters = VecDeque::new();
        let active_adapters = VecDeque::new();
        let tracked_adapters = HashSet::new();

        Self {
            queue_map,
            pending_adapters,
            active_adapters,
            tracked_adapters,
            max_active_adapters: max_active_adapters,
            memory_budget_remaining: 1.0,
            max_active_time: Duration::from_secs(adapter_cycle_time_s),
            next_id: 0,
        }
    }

    /// Append entry to the appropriate queue
    pub(crate) fn append(
        &mut self,
        adapter: Adapter,
        adapter_event: Arc<AdapterEvent>,
        entry: Entry,
    ) -> bool {
        // check if queue_map has adapter_key as key
        // if not, then add a new Queue and download the adapter
        let mut download = false;
        if !self.queue_map.contains_key(&adapter) {
            self.queue_map.insert(
                adapter.clone(),
                QueueState::new(adapter.clone(), adapter_event.clone()),
            );
            download = true;
        }

        if !self.tracked_adapters.contains(&adapter) {
            self.tracked_adapters.insert(adapter.clone());
            self.pending_adapters.push_back(adapter.clone());
        }

        // ensure that append completes before sending batcher message
        let queue = self.queue_map.get_mut(&adapter).unwrap();
        queue.append(self.next_id, entry);
        self.next_id += 1;

        return download;
    }

    /// Removes adapter queue from the map
    pub(crate) fn remove(&mut self, adapter: &Adapter) {
        self.queue_map.remove(adapter);
        self.active_adapters.retain(|id| id != adapter);
        self.pending_adapters.retain(|id| id != adapter);
        self.tracked_adapters.remove(&adapter);
    }

    /// Removes the adapter queue from the tracked set and its queues
    pub(crate) fn untrack(&mut self, adapter: &Adapter) {
        self.active_adapters.retain(|id| id != adapter);
        self.pending_adapters.retain(|id| id != adapter);
        self.tracked_adapters.remove(&adapter);
    }

    pub(crate) fn has_adapter(&self, adapter: &Adapter) -> bool {
        self.queue_map.contains_key(adapter)
    }

    /// Get any queues that are in an errored state
    pub(crate) fn get_errored_adapters(&mut self) -> Vec<Adapter> {
        let mut errored_adapters = Vec::new();
        for (adapter, queue) in self.queue_map.iter() {
            if queue.status() == &AdapterStatus::Errored {
                errored_adapters.push(adapter.clone());
            }
        }
        errored_adapters
    }

    pub(crate) fn set_cost(&mut self, adapter: &Adapter, cost: f32) {
        let q = self.queue_map.get_mut(adapter);
        if q.is_none() {
            // TODO(travis): remove this
            tracing::error!("adapter {} not found in queue_map", adapter.as_string());
            println!("{:?}", Backtrace::force_capture());
        }
        let queue = q.unwrap();
        queue.set_cost(cost);
    }

    pub(crate) fn set_status(&mut self, adapter: &Adapter, status: AdapterStatus) {
        let q = self.queue_map.get_mut(adapter);
        if q.is_none() {
            // TODO(travis): remove this
            tracing::error!("adapter {} not found in queue_map", adapter.as_string());
            println!("{:?}", Backtrace::force_capture());
        }
        let queue = q.unwrap();
        queue.set_status(status);
    }

    pub(crate) fn push_front(&mut self, adapter: &Adapter, entry_id: u64, entry: Entry) {
        let queue = self.queue_map.get_mut(adapter).unwrap();
        queue.push_front(entry_id, entry);
    }

    pub(crate) fn drain(
        &mut self,
        adapter: &Adapter,
    ) -> std::collections::vec_deque::Drain<(u64, Entry)> {
        let queue = self.queue_map.get_mut(adapter).unwrap();
        queue.drain()
    }

    pub(crate) fn len(&self) -> usize {
        self.queue_map
            .values()
            .map(|queue| queue.entries().len())
            .sum()
    }

    pub(crate) fn active_len(&self) -> usize {
        self.active_adapters.len()
    }

    fn get_oldest_active_adapter(&mut self) -> Option<Adapter> {
        // Returns the adapter that maps to the queue whose front entry has the oldest activation timestamp,
        // but prefer queues that have not ben active past the maximum time limit
        let now = Instant::now();
        let mut oldest_adapter = None;
        let mut oldest_ts = Instant::now();
        let mut oldest_within_limit_adapter = None;
        let mut oldest_within_limit_ts = Instant::now();
        for adapter in self.active_adapters.iter() {
            let queue = self.queue_map.get(adapter).unwrap();
            if queue.is_empty() || queue.status() != &AdapterStatus::Ready {
                continue;
            }

            if let Some(ts) = queue.peek() {
                if ts < oldest_ts {
                    oldest_ts = ts;
                    oldest_adapter = Some(adapter.clone());
                }

                if ts < oldest_within_limit_ts
                    && now.duration_since(queue.activation_ts().unwrap()) < self.max_active_time
                {
                    oldest_within_limit_ts = ts;
                    oldest_within_limit_adapter = Some(adapter.clone());
                }
            }
        }

        // Return the oldest adapter whose queue has been active for less than the limit if it exists,
        // otherwise return the oldest adapter across all queues
        if oldest_within_limit_adapter.is_some() {
            oldest_within_limit_adapter
        } else {
            oldest_adapter
        }
    }

    /// Update the queues of pending and active adapters based on the current state
    pub(crate) fn update_adapters(
        &mut self,
        adapters_in_use: &HashSet<Adapter>,
    ) -> (Vec<Adapter>, Vec<Adapter>) {
        let mut offload_adapters = Vec::new();
        let mut load_adapters = Vec::new();

        // Mark any active adapters that are Idle (have no active or pending requests) for removal
        // Additionally, move any adapters that have been activate over the limit to pending
        let now = Instant::now();
        let mut adapters_to_remove = HashSet::new();
        for adapter in self.active_adapters.iter() {
            let queue = self.queue_map.get(adapter).unwrap();
            if adapters_in_use.contains(&queue.adapter()) {
                // Cannot modify active adapters that are in use
                continue;
            }

            if self.pending_adapters.len() <= adapters_to_remove.len() {
                // Only move adapters out of active if we have pending adapters ready to take their place
                continue;
            }

            if queue.entries().is_empty() {
                // queue is empty and not in use, so move to removal set
                adapters_to_remove.insert(adapter.clone());

                // Start async offload process
                // TODO(travis): we're being too aggressive about offloading here, we should only
                // add adapters to this set if the number of active adapters is full and there are new adapters
                // waiting to be loaded
                offload_adapters.push(adapter.clone());
            }
        }

        // Remove all adapters in the remove set
        self.active_adapters
            .retain(|adapter| !adapters_to_remove.contains(adapter));
        self.tracked_adapters
            .retain(|adapter| !adapters_to_remove.contains(adapter));

        // Move the front adapter from the active set if it has been active over the limit to pending.
        // Do this after filtering out idle adapters as those should take priority over adapters that
        // have been active over the limit.
        if !self.active_adapters.is_empty() {
            let adapter = self.active_adapters.front().unwrap().clone();
            let queue = self.queue_map.get(&adapter).unwrap();
            if !adapters_in_use.contains(&queue.adapter())
                && now.duration_since(queue.activation_ts().unwrap()) > self.max_active_time
                && self.pending_adapters.len() >= 1
            {
                self.active_adapters.pop_front();
                self.pending_adapters.push_back(adapter.clone());

                // Start async offload process
                offload_adapters.push(adapter.clone());
            }
        }

        // Add back cost for all offload adapters
        for adapter in offload_adapters.iter() {
            let queue = self.queue_map.get(adapter).unwrap();
            let cost = queue.cost().unwrap();
            self.memory_budget_remaining += cost;
            tracing::info!(
                "offloading adapter {} with cost {} (memory budget remaining: {})",
                adapter.as_string(),
                cost,
                self.memory_budget_remaining
            );
        }

        // Add pending adapters to the active set until we reach the max
        while self.active_adapters.len() < self.max_active_adapters
            && self.pending_adapters.len() > 0
        {
            let queue = self
                .queue_map
                .get_mut(self.pending_adapters.front().unwrap())
                .unwrap();
            if queue.cost().is_none() {
                // Adapter has not been downloaded yet
                break;
            }

            // Check to see that we have enough memory budget remaining to load the adapter
            let cost = queue.cost().unwrap();
            if cost > self.memory_budget_remaining {
                // Adapter is too expensive to load
                break;
            }

            // Update activation timestamp
            let adapter = self.pending_adapters.pop_front().unwrap();
            queue.set_activation_ts(now);

            // Calculate remaining memory budget
            self.memory_budget_remaining -= cost;

            // Start async loading process
            load_adapters.push(adapter.clone());

            self.active_adapters.push_back(adapter.clone());

            tracing::info!(
                "loading adapter {} with cost {} (memory budget remaining: {})",
                adapter.as_string(),
                cost,
                self.memory_budget_remaining
            );
        }

        (offload_adapters, load_adapters)
    }

    pub(crate) fn next_entry(&mut self) -> Option<(u64, Entry, Adapter)> {
        // Get the adapter from the active set that has been waiting the longest.
        let adapter = self.get_oldest_active_adapter();
        if adapter.is_none() {
            // No active adapter has any entries
            tracing::debug!("No active adapter has any entries");
            return None;
        }

        // Pop the oldest entry from the queue
        let adapter = adapter.unwrap();
        let queue = self.queue_map.get_mut(&adapter).unwrap();
        let (id, entry, _next_oldest_entry) = queue.pop().unwrap();
        Some((id, entry, adapter))
    }
}
