//! NUMA-aware memory allocation and thread affinity.
//!
//! Distributes expert weights across NUMA nodes so each expert's data is
//! local to the threads that compute it. On NPS4 (EPYC 7742), this avoids
//! cross-node memory traffic which costs ~2x latency vs local access.
//!
//! Falls back gracefully to standard allocation when:
//! - libnuma is not available
//! - System has only 1 NUMA node (NPS1)
//! - NUMA allocation fails for any reason

use std::collections::HashMap;

/// NUMA topology information.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes visible to the OS.
    pub num_nodes: usize,
    /// CPUs (logical core IDs) on each node.
    pub node_cpus: Vec<Vec<usize>>,
    /// Available memory per node in bytes.
    pub node_mem_bytes: Vec<u64>,
}

/// A memory region allocated on a specific NUMA node.
/// Freed via `numa_free` on drop.
pub struct NumaAlloc {
    ptr: *mut u8,
    len: usize,
    node: usize,
}

// SAFETY: NumaAlloc owns its memory and doesn't share it.
unsafe impl Send for NumaAlloc {}
unsafe impl Sync for NumaAlloc {}

impl Drop for NumaAlloc {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                numa_free(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

impl NumaAlloc {
    /// Get the raw pointer to the allocated memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get a mutable pointer to the allocated memory.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Length of the allocation in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Which NUMA node this allocation is on.
    pub fn node(&self) -> usize {
        self.node
    }

    /// Get a typed slice view of the allocation.
    ///
    /// # Safety
    /// Caller must ensure the data has been properly initialized as type T.
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        let count = self.len / std::mem::size_of::<T>();
        std::slice::from_raw_parts(self.ptr as *const T, count)
    }

    /// Get a mutable typed slice view of the allocation.
    ///
    /// # Safety
    /// Caller must ensure proper alignment and initialization.
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let count = self.len / std::mem::size_of::<T>();
        std::slice::from_raw_parts_mut(self.ptr as *mut T, count)
    }
}

// ── libnuma FFI ─────────────────────────────────────────────────────
//
// When built with libnuma (default on Linux with libnuma-dev installed),
// we link directly. When built without (CI, minimal containers), the
// cfg(no_numa) stubs return "not available" so everything falls back
// to standard allocation.

#[cfg(not(no_numa))]
extern "C" {
    fn numa_available() -> libc::c_int;
    fn numa_max_node() -> libc::c_int;
    fn numa_alloc_onnode(size: libc::size_t, node: libc::c_int) -> *mut libc::c_void;
    fn numa_free(start: *mut libc::c_void, size: libc::size_t);
    fn numa_node_of_cpu(cpu: libc::c_int) -> libc::c_int;
    fn numa_num_configured_cpus() -> libc::c_int;
    fn numa_run_on_node(node: libc::c_int) -> libc::c_int;
}

// Stubs when libnuma is not available at build time
#[cfg(no_numa)]
unsafe fn numa_available() -> libc::c_int { -1 }
#[cfg(no_numa)]
unsafe fn numa_max_node() -> libc::c_int { 0 }
#[cfg(no_numa)]
unsafe fn numa_alloc_onnode(_size: libc::size_t, _node: libc::c_int) -> *mut libc::c_void { std::ptr::null_mut() }
#[cfg(no_numa)]
unsafe fn numa_free(_start: *mut libc::c_void, _size: libc::size_t) {}
#[cfg(no_numa)]
unsafe fn numa_node_of_cpu(_cpu: libc::c_int) -> libc::c_int { 0 }
#[cfg(no_numa)]
unsafe fn numa_num_configured_cpus() -> libc::c_int { 1 }
#[cfg(no_numa)]
unsafe fn numa_run_on_node(_node: libc::c_int) -> libc::c_int { -1 }

/// Check if libnuma is available and functional.
fn numa_is_available() -> bool {
    unsafe { numa_available() >= 0 }
}

impl NumaTopology {
    /// Detect the system's NUMA topology.
    /// Returns a topology with 1 node if NUMA is not available.
    pub fn detect() -> Self {
        if !numa_is_available() {
            log::info!("NUMA: libnuma not available, using single-node fallback");
            return Self::single_node();
        }

        let max_node = unsafe { numa_max_node() } as usize;
        let num_nodes = max_node + 1;
        let num_cpus = unsafe { numa_num_configured_cpus() } as usize;

        // Map CPUs to nodes
        let mut node_cpus: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for cpu in 0..num_cpus {
            let node = unsafe { numa_node_of_cpu(cpu as libc::c_int) };
            if node >= 0 && (node as usize) < num_nodes {
                node_cpus[node as usize].push(cpu);
            }
        }

        // Read available memory per node from sysfs
        let mut node_mem_bytes = vec![0u64; num_nodes];
        for node in 0..num_nodes {
            let meminfo_path = format!("/sys/devices/system/node/node{node}/meminfo");
            if let Ok(content) = std::fs::read_to_string(&meminfo_path) {
                for line in content.lines() {
                    if line.contains("MemFree:") || line.contains("MemAvailable:") {
                        if let Some(kb) = line.split_whitespace().nth(3) {
                            if let Ok(val) = kb.parse::<u64>() {
                                node_mem_bytes[node] = val * 1024;
                                break;
                            }
                        }
                    }
                }
            }
        }

        log::info!(
            "NUMA: {} nodes, {} total CPUs",
            num_nodes, num_cpus,
        );
        for (i, cpus) in node_cpus.iter().enumerate() {
            log::info!(
                "  Node {}: {} CPUs, {:.1} GB free",
                i,
                cpus.len(),
                node_mem_bytes[i] as f64 / 1e9,
            );
        }

        NumaTopology {
            num_nodes,
            node_cpus,
            node_mem_bytes,
        }
    }

    /// Fallback for systems without NUMA or with only 1 node.
    fn single_node() -> Self {
        let num_cpus = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } as usize;
        NumaTopology {
            num_nodes: 1,
            node_cpus: vec![(0..num_cpus).collect()],
            node_mem_bytes: vec![0], // unknown
        }
    }

    /// Whether NUMA-aware placement is meaningful (>1 node).
    pub fn is_numa(&self) -> bool {
        self.num_nodes > 1
    }
}

/// Allocate memory on a specific NUMA node.
/// Falls back to standard allocation if NUMA is not available.
pub fn alloc_on_node(size: usize, node: usize) -> Option<NumaAlloc> {
    if !numa_is_available() || size == 0 {
        return None;
    }

    let ptr = unsafe { numa_alloc_onnode(size, node as libc::c_int) };
    if ptr.is_null() {
        log::warn!("numa_alloc_onnode({size} bytes, node {node}) failed");
        return None;
    }

    Some(NumaAlloc {
        ptr: ptr as *mut u8,
        len: size,
        node,
    })
}

/// Assignment of experts to NUMA nodes.
#[derive(Debug, Clone)]
pub struct NumaExpertMap {
    /// For each MoE layer, for each expert: which NUMA node it's assigned to.
    /// Indexed as [moe_layer_idx][expert_idx] → node_id.
    pub assignments: Vec<Vec<usize>>,
    /// Number of NUMA nodes.
    pub num_nodes: usize,
}

impl NumaExpertMap {
    /// Create a round-robin assignment of experts to NUMA nodes.
    /// Expert i in each layer goes to node (i % num_nodes).
    pub fn round_robin(num_moe_layers: usize, num_experts: usize, num_nodes: usize) -> Self {
        let assignments: Vec<Vec<usize>> = (0..num_moe_layers)
            .map(|_| {
                (0..num_experts)
                    .map(|eidx| eidx % num_nodes)
                    .collect()
            })
            .collect();

        NumaExpertMap {
            assignments,
            num_nodes,
        }
    }

    /// Get the NUMA node for a specific expert.
    pub fn node_for(&self, moe_layer_idx: usize, expert_idx: usize) -> usize {
        self.assignments[moe_layer_idx][expert_idx]
    }

    /// Get all experts on a given node for a given layer, sorted by expert index.
    pub fn experts_on_node(&self, moe_layer_idx: usize, node: usize) -> Vec<usize> {
        self.assignments[moe_layer_idx]
            .iter()
            .enumerate()
            .filter(|(_, &n)| n == node)
            .map(|(eidx, _)| eidx)
            .collect()
    }

    /// Sort expert indices by NUMA node, returning (expert_idx, node) pairs.
    /// This enables running all node-0 experts first, then node-1, etc.
    pub fn sort_by_node(&self, moe_layer_idx: usize, expert_indices: &[usize]) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = expert_indices
            .iter()
            .map(|&eidx| (eidx, self.node_for(moe_layer_idx, eidx)))
            .collect();
        pairs.sort_by_key(|&(_, node)| node);
        pairs
    }

    /// Group expert indices by NUMA node.
    /// Returns a map from node_id → Vec<(original_position, expert_idx)>.
    pub fn group_by_node(
        &self,
        moe_layer_idx: usize,
        expert_indices: &[usize],
    ) -> HashMap<usize, Vec<(usize, usize)>> {
        let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (pos, &eidx) in expert_indices.iter().enumerate() {
            let node = self.node_for(moe_layer_idx, eidx);
            groups.entry(node).or_default().push((pos, eidx));
        }
        groups
    }
}

// ── Page migration via mbind ────────────────────────────────────────

extern "C" {
    fn mbind(
        addr: *mut libc::c_void,
        len: libc::size_t,
        mode: libc::c_int,
        nodemask: *const libc::c_ulong,
        maxnode: libc::c_ulong,
        flags: libc::c_uint,
    ) -> libc::c_int;
}

const MPOL_BIND: libc::c_int = 2;
const MPOL_MF_MOVE: libc::c_uint = 2;

/// Move existing memory pages to a specific NUMA node via mbind().
/// Works on any allocated memory — aligns to page boundaries automatically.
/// Returns true on success.
pub fn migrate_to_node(ptr: *mut u8, len: usize, node: usize) -> bool {
    if !numa_is_available() || len == 0 || node >= 64 {
        return false;
    }

    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    // Align down to page boundary
    let aligned_addr = (ptr as usize) & !(page_size - 1);
    let offset = (ptr as usize) - aligned_addr;
    let aligned_len = len + offset;

    let nodemask: u64 = 1u64 << node;

    let ret = unsafe {
        mbind(
            aligned_addr as *mut libc::c_void,
            aligned_len,
            MPOL_BIND,
            &nodemask as *const u64 as *const libc::c_ulong,
            64, // maxnode (supports up to 64 nodes)
            MPOL_MF_MOVE,
        )
    };

    if ret != 0 {
        let err = std::io::Error::last_os_error();
        log::debug!("mbind to node {node} failed: {err}");
        return false;
    }
    true
}

/// Migrate a Vec's backing memory to a specific NUMA node.
/// Returns true if the migration was successful.
pub fn migrate_vec_to_node<T>(vec: &mut Vec<T>, node: usize) -> bool {
    if vec.is_empty() {
        return true;
    }
    let ptr = vec.as_mut_ptr() as *mut u8;
    let len = vec.len() * std::mem::size_of::<T>();
    migrate_to_node(ptr, len, node)
}

/// Pin the current thread to a specific NUMA node.
/// This restricts the thread to CPUs on the given node.
pub fn pin_thread_to_node(node: usize) -> bool {
    if !numa_is_available() {
        return false;
    }
    let result = unsafe { numa_run_on_node(node as libc::c_int) };
    result == 0
}

/// Reset thread affinity to allow running on any node.
pub fn unpin_thread() -> bool {
    if !numa_is_available() {
        return false;
    }
    // node=-1 means "run on any node"
    let result = unsafe { numa_run_on_node(-1) };
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_detection() {
        let topo = NumaTopology::detect();
        eprintln!("NUMA topology: {:?}", topo);
        assert!(topo.num_nodes >= 1);
        assert!(!topo.node_cpus.is_empty());
        // Total CPUs across all nodes should be > 0
        let total_cpus: usize = topo.node_cpus.iter().map(|v| v.len()).sum();
        assert!(total_cpus > 0);
        eprintln!(
            "  {} nodes, {} total CPUs, is_numa={}",
            topo.num_nodes, total_cpus, topo.is_numa(),
        );
    }

    #[test]
    fn test_expert_map_round_robin() {
        let map = NumaExpertMap::round_robin(3, 8, 4);

        // Expert 0 → node 0, expert 1 → node 1, ..., expert 4 → node 0, etc.
        assert_eq!(map.node_for(0, 0), 0);
        assert_eq!(map.node_for(0, 1), 1);
        assert_eq!(map.node_for(0, 3), 3);
        assert_eq!(map.node_for(0, 4), 0);
        assert_eq!(map.node_for(0, 7), 3);

        // Sort by node
        let sorted = map.sort_by_node(0, &[3, 0, 7, 4, 1, 5, 2, 6]);
        // Node 0: experts 0, 4  Node 1: experts 1, 5  Node 2: experts 2, 6  Node 3: experts 3, 7
        assert_eq!(sorted[0].1, 0); // first two should be node 0
        assert_eq!(sorted[1].1, 0);
        assert_eq!(sorted[2].1, 1);
        assert_eq!(sorted[3].1, 1);

        // Group by node
        let groups = map.group_by_node(0, &[0, 1, 4, 7]);
        assert_eq!(groups[&0].len(), 2); // experts 0, 4
        assert_eq!(groups[&1].len(), 1); // expert 1
        assert_eq!(groups[&3].len(), 1); // expert 7
    }

    #[test]
    fn test_numa_alloc() {
        let topo = NumaTopology::detect();
        if !topo.is_numa() {
            eprintln!("Single NUMA node — testing node 0 allocation");
        }

        // Try allocating 1 MB on node 0
        let alloc = alloc_on_node(1024 * 1024, 0);
        match alloc {
            Some(mut a) => {
                assert_eq!(a.len(), 1024 * 1024);
                assert_eq!(a.node(), 0);
                // Write and read back
                unsafe {
                    let slice = a.as_mut_slice::<u8>();
                    slice[0] = 42;
                    slice[1024 * 1024 - 1] = 99;
                    assert_eq!(a.as_slice::<u8>()[0], 42);
                    assert_eq!(a.as_slice::<u8>()[1024 * 1024 - 1], 99);
                }
                eprintln!("NUMA alloc on node 0: {} bytes OK", a.len());
            }
            None => {
                eprintln!("NUMA alloc not available (libnuma missing or not functional)");
            }
        }
    }

    #[test]
    fn test_thread_pinning() {
        let topo = NumaTopology::detect();
        if !topo.is_numa() {
            eprintln!("Single NUMA node — pinning test limited");
        }

        // Pin to node 0
        let pinned = pin_thread_to_node(0);
        eprintln!("Pin to node 0: {}", if pinned { "OK" } else { "N/A" });

        // Unpin
        let unpinned = unpin_thread();
        eprintln!("Unpin: {}", if unpinned { "OK" } else { "N/A" });
    }
}
