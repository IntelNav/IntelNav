//! In-memory registry state.
//!
//! One `RegistryState` holds the manifest plus the live seeder table. All
//! mutating calls go through `&self` (interior `RwLock`) so the axum
//! handlers can be `Arc`-shared cheaply.
//!
//! This is v1 (single-node, in-memory). Persistence and crash recovery are
//! intentionally out of scope — if the registry restarts, peers re-claim on
//! their next heartbeat cycle (~20 s).

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use intelnav_core::Role;

use crate::manifest::{Manifest, ManifestPart};

/// What the registry wants a seeder to do on its next heartbeat response.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Directive {
    Standby,
    Resume,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SeederStatus {
    Live,
    Standby,
    Draining,
    Evicted,
}

#[derive(Clone, Debug, Serialize)]
pub struct SeederEntry {
    pub peer_id:        String,   // 64-char hex
    pub role:           Role,
    pub joined_at:      i64,      // unix secs
    pub last_heartbeat: i64,
    pub status:         SeederStatus,
    #[serde(skip)]
    pub pending_directive: Option<Directive>,
    pub service_credits:   u64,   // spec §6: stub counter, no economic effect
}

#[derive(Clone, Debug, Serialize)]
pub struct Reservation {
    pub peer_id:    String,
    pub part_id:    String,
    pub expires_at: i64,
}

#[derive(Debug)]
struct Inner {
    manifest: Manifest,
    // part_id → seeders (peer_id → entry)
    seeders:  HashMap<String, HashMap<String, SeederEntry>>,
    // peer_id → outstanding reservation
    reservations: HashMap<String, Reservation>,
    // peer_id → sliding-window counter of assign() calls
    assign_calls: HashMap<String, Vec<i64>>,
}

pub struct RegistryState {
    inner: RwLock<Inner>,
}

impl RegistryState {
    pub fn new(manifest: Manifest) -> Self {
        let mut seeders = HashMap::new();
        for p in &manifest.parts {
            seeders.insert(p.id.clone(), HashMap::new());
        }
        Self {
            inner: RwLock::new(Inner {
                manifest,
                seeders,
                reservations: HashMap::new(),
                assign_calls: HashMap::new(),
            }),
        }
    }

    pub fn manifest_snapshot(&self) -> Manifest {
        self.inner.read().unwrap().manifest.clone()
    }

    // ------------------------------------------------------------------
    //  queries
    // ------------------------------------------------------------------

    /// Spec §3.4: live + sufficiently-aged volunteers for this part.
    pub fn live_volunteers(&self, part_id: &str, now: i64) -> usize {
        let inner = self.inner.read().unwrap();
        let min_live = inner.manifest.defaults.min_live_seconds as i64;
        inner
            .seeders
            .get(part_id)
            .map(|m| {
                m.values()
                    .filter(|s| s.role == Role::Volunteer
                        && s.status == SeederStatus::Live
                        && now - s.joined_at >= min_live)
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn seeders_of(&self, part_id: &str) -> Vec<SeederEntry> {
        self.inner
            .read()
            .unwrap()
            .seeders
            .get(part_id)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn replication_counts(&self) -> HashMap<String, (usize, usize)> {
        // part_id → (live_volunteers_counted, live_total)
        let inner = self.inner.read().unwrap();
        let now = unix_now();
        let min_live = inner.manifest.defaults.min_live_seconds as i64;
        let mut out = HashMap::new();
        for (part_id, seeders) in &inner.seeders {
            let mut counted = 0usize;
            let mut live = 0usize;
            for s in seeders.values() {
                if s.status == SeederStatus::Live { live += 1; }
                if s.role == Role::Volunteer
                    && s.status == SeederStatus::Live
                    && now - s.joined_at >= min_live
                {
                    counted += 1;
                }
            }
            out.insert(part_id.clone(), (counted, live));
        }
        out
    }

    // ------------------------------------------------------------------
    //  assign / claim / heartbeat / release
    // ------------------------------------------------------------------

    pub fn try_assign(
        &self,
        peer_id_hex: &str,
        role: Role,
        vram_bytes: u64,
    ) -> Result<AssignmentOut, AssignError> {
        let mut inner = self.inner.write().unwrap();
        let now = unix_now();

        // rate limit (§4.3)
        if role == Role::Volunteer {
            let per_hour = inner.manifest.defaults.assign_rate_per_hour as usize;
            let window = 3600;
            let entry = inner.assign_calls.entry(peer_id_hex.to_string()).or_default();
            entry.retain(|t| now - *t < window);
            if entry.len() >= per_hour {
                return Err(AssignError::RateLimited);
            }
            entry.push(now);
        }

        // purge expired reservations first
        inner.reservations.retain(|_, r| r.expires_at > now);

        // Pick the part with the smallest (counted_volunteers + outstanding
        // reservations). VRAM filter applies first.
        let min_live = inner.manifest.defaults.min_live_seconds as i64;
        let ttl = inner.manifest.defaults.reservation_ttl_s as i64;
        let manifest = inner.manifest.clone();
        let mut best: Option<(&ManifestPart, usize)> = None;
        for p in &manifest.parts {
            if p.min_vram_bytes > 0 && vram_bytes > 0 && vram_bytes < p.min_vram_bytes {
                continue;
            }
            let live_counted = inner.seeders.get(&p.id)
                .map(|m| m.values().filter(|s|
                    s.role == Role::Volunteer
                    && s.status == SeederStatus::Live
                    && now - s.joined_at >= min_live).count())
                .unwrap_or(0);
            let reserved = inner.reservations.values()
                .filter(|r| r.part_id == p.id).count();
            let score = live_counted + reserved;
            match &best {
                None => best = Some((p, score)),
                Some((_, s)) if score < *s => best = Some((p, score)),
                _ => {}
            }
        }
        let (part, _) = best.ok_or(AssignError::NoPartFits)?;

        // reservation
        let expires_at = now + ttl;
        inner.reservations.insert(
            peer_id_hex.to_string(),
            Reservation { peer_id: peer_id_hex.to_string(), part_id: part.id.clone(), expires_at },
        );

        Ok(AssignmentOut {
            part_id:            part.id.clone(),
            layer_range:        part.layer_range,
            weight_url:         part.weight_url.clone(),
            sha256:             part.sha256.clone(),
            size_bytes:         part.size_bytes,
            reservation_ttl_s:  inner.manifest.defaults.reservation_ttl_s,
        })
    }

    pub fn try_claim(
        &self,
        part_id: &str,
        peer_id_hex: &str,
        role: Role,
    ) -> Result<(), ClaimError> {
        let mut inner = self.inner.write().unwrap();
        let now = unix_now();

        if inner.manifest.part(part_id).is_none() {
            return Err(ClaimError::UnknownPart);
        }

        let seeders = inner.seeders.entry(part_id.to_string()).or_default();
        if let Some(existing) = seeders.get(peer_id_hex) {
            if matches!(existing.status, SeederStatus::Live | SeederStatus::Draining) {
                return Err(ClaimError::AlreadyLive);
            }
        }

        seeders.insert(peer_id_hex.to_string(), SeederEntry {
            peer_id:        peer_id_hex.to_string(),
            role,
            joined_at:      now,
            last_heartbeat: now,
            status:         SeederStatus::Live,
            pending_directive: None,
            service_credits:   0,
        });

        // claim consumes any outstanding reservation
        inner.reservations.remove(peer_id_hex);

        self.recompute_directives_locked(&mut inner, now);
        Ok(())
    }

    pub fn try_heartbeat(
        &self,
        part_id: &str,
        peer_id_hex: &str,
    ) -> Result<HeartbeatOut, HeartbeatError> {
        let mut inner = self.inner.write().unwrap();
        let now = unix_now();

        let Some(seeders) = inner.seeders.get_mut(part_id) else {
            return Err(HeartbeatError::UnknownPart);
        };
        let Some(entry) = seeders.get_mut(peer_id_hex) else {
            return Err(HeartbeatError::NotLive);
        };
        if entry.status == SeederStatus::Evicted {
            return Err(HeartbeatError::NotLive);
        }

        entry.last_heartbeat = now;
        entry.service_credits = entry.service_credits.saturating_add(1);
        let directive = entry.pending_directive.take();

        // apply directive locally
        if let Some(d) = directive {
            match d {
                Directive::Standby => entry.status = SeederStatus::Standby,
                Directive::Resume  => entry.status = SeederStatus::Live,
            }
        }

        // eviction sweep + directive recompute happen on every hb cycle
        self.evict_stale_locked(&mut inner, now);
        self.recompute_directives_locked(&mut inner, now);

        Ok(HeartbeatOut { directive })
    }

    pub fn try_release(
        &self,
        part_id: &str,
        peer_id_hex: &str,
    ) -> Result<(), ReleaseError> {
        let mut inner = self.inner.write().unwrap();
        let now = unix_now();
        let Some(seeders) = inner.seeders.get_mut(part_id) else {
            return Err(ReleaseError::UnknownPart);
        };
        let Some(entry) = seeders.get_mut(peer_id_hex) else {
            return Err(ReleaseError::NotLive);
        };
        entry.status = SeederStatus::Draining;
        self.recompute_directives_locked(&mut inner, now);
        Ok(())
    }

    // ------------------------------------------------------------------
    //  internal: eviction + directive computation
    // ------------------------------------------------------------------

    fn evict_stale_locked(&self, inner: &mut Inner, now: i64) {
        let hb_interval = inner.manifest.defaults.heartbeat_interval_s as i64;
        let miss_tol    = inner.manifest.defaults.heartbeat_miss_tolerance as i64;
        let limit       = hb_interval * miss_tol;
        for seeders in inner.seeders.values_mut() {
            for entry in seeders.values_mut() {
                if entry.status == SeederStatus::Evicted { continue; }
                if now - entry.last_heartbeat > limit {
                    entry.status = SeederStatus::Evicted;
                }
            }
        }
    }

    /// Apply the standby/resume hysteresis rule (§3.4).
    fn recompute_directives_locked(&self, inner: &mut Inner, now: i64) {
        let min_live = inner.manifest.defaults.min_live_seconds as i64;
        let manifest = inner.manifest.clone();
        for part in &manifest.parts {
            let desired_k = manifest.desired_k_of(part) as usize;
            let Some(seeders) = inner.seeders.get_mut(&part.id) else { continue; };
            let live_v = seeders.values()
                .filter(|s| s.role == Role::Volunteer
                    && s.status == SeederStatus::Live
                    && now - s.joined_at >= min_live)
                .count();
            for s in seeders.values_mut() {
                if s.role != Role::Cloud { continue; }
                match s.status {
                    SeederStatus::Live => {
                        if live_v >= desired_k {
                            s.pending_directive = Some(Directive::Standby);
                        }
                    }
                    SeederStatus::Standby => {
                        if live_v + 1 <= desired_k {   // live_v ≤ k-1
                            s.pending_directive = Some(Directive::Resume);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Periodic maintenance tick, called by `server::reaper`.
    pub fn tick(&self) {
        let mut inner = self.inner.write().unwrap();
        let now = unix_now();
        self.evict_stale_locked(&mut inner, now);
        self.recompute_directives_locked(&mut inner, now);
    }
}

// ----------------------------------------------------------------------
//  result types
// ----------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct AssignmentOut {
    pub part_id:            String,
    pub layer_range:        [u16; 2],
    pub weight_url:         String,
    pub sha256:             String,
    pub size_bytes:         u64,
    pub reservation_ttl_s:  u32,
}

#[derive(Debug, thiserror::Error)]
pub enum AssignError {
    #[error("rate limited")]       RateLimited,
    #[error("no part fits this peer's capabilities")] NoPartFits,
}

#[derive(Clone, Debug, Serialize)]
pub struct HeartbeatOut {
    pub directive: Option<Directive>,
}

#[derive(Debug, thiserror::Error)]
pub enum ClaimError {
    #[error("unknown part")]   UnknownPart,
    #[error("already live")]   AlreadyLive,
}

#[derive(Debug, thiserror::Error)]
pub enum HeartbeatError {
    #[error("unknown part")] UnknownPart,
    #[error("peer not in live set")] NotLive,
}

#[derive(Debug, thiserror::Error)]
pub enum ReleaseError {
    #[error("unknown part")] UnknownPart,
    #[error("peer not in live set")] NotLive,
}

// ----------------------------------------------------------------------

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
