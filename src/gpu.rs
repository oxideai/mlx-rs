/// Properties to control the GPU memory allocation and buffer reuse.
///
/// [active_memory()] + [cache_memory()] is the total memory allocated by MLX.
/// [active_memory()] is in currently active [Array] and [cache_memory()]
/// is recently used memory that can be recycled.
///
/// Control the size of [cache_memory()] via [set_cache_limit()].
/// and the overall memory limit with [set_memory_limit()].
///
/// Examine memory use over time with [snapshot()] and [Snapshot].

static mut CACHE_LIMIT: Option<usize> = None;
static mut MEMORY_LIMIT: Option<usize> = None;
static mut RELAXED_MEMORY_LIMIT: bool = true;

/// Snapshot of memory stats.
///
/// [active_memory()] + [cache_memory()] is the total memory allocated by MLX.
/// [active_memory()] is in currently active [Array] and [cache_memory()]
/// is recently used memory that can be recycled.
///
/// Control the size of [active_memory()] via [set_cache_limit()].
/// and the overall memory limit with [set_memory_limit()].
///
/// This might be used to examine memory use over a run or sample it during a run:
///
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    /// See [active_memory()]
    pub active_memory: usize,
    /// See [cache_memory()]
    pub cache_memory: usize,
    /// See [peak_memory()]
    pub peak_memory: usize,
}

impl Snapshot {
    /// Compute the difference between two snapshots:
    ///
    /// ```rust
    /// use mlx::gpu;
    /// let start_memory = gpu::snapshot();
    /// //...
    /// let end_memory = gpu::snapshot();
    /// println!("{}" ,start_memory.delta(&end_memory));
    /// ```
    pub fn delta(&self, other: &Snapshot) -> Snapshot {
        Snapshot {
            active_memory: other.active_memory - self.active_memory,
            cache_memory: other.cache_memory - self.cache_memory,
            peak_memory: other.peak_memory - self.peak_memory,
        }
    }
}

impl std::fmt::Display for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        fn scale(value: i32, width: i32) -> String {
            let value = if value > 1024 * 1024 * 10 {
                format!("{}M", value / (1024 * 1024))
            } else {
                format!("{}K", value / 1024)
            };

            let pad = std::cmp::max(0, width - value.len() as i32);
            format!("{}{}", value, " ".repeat(pad as usize))
        }

        write!(
            f,
            r#"
            Peak:   {} ({})
            Active: {} ({})
            Cache:  {} ({})
            "#,
            scale(self.peak_memory as i32, 12),
            self.peak_memory,
            scale(self.active_memory as i32, 12),
            self.active_memory,
            scale(self.cache_memory as i32, 12),
            self.cache_memory
        )
    }
}

/// Get the actively used memory in bytes.
///
/// Note, this will not always match memory use reported by the system because
/// it does not include cached memory buffers.
pub fn active_memory() -> usize {
    unsafe { mlx_sys::mlx_metal_get_active_memory() }
}

/// Get the cache size in bytes.
///
/// The cache includes memory not currently used that has not been returned
/// to the system allocator.
pub fn cache_memory() -> usize {
    unsafe { mlx_sys::mlx_metal_get_cache_memory() }
}

/// Get the peak amount of active memory in bytes.
///
/// The maximum memory used is recorded from the beginning of the program
/// execution.
pub fn peak_memory() -> usize {
    unsafe { mlx_sys::mlx_metal_get_peak_memory() }
}

/// Return a snapshot of memory stats -- see [Snapshot] for more details.
///
/// Get the current memory use.  This can be used to measure before/after and current memory use:
///
/// ```rust
/// use mlx::gpu;
/// let current_memory = gpu::snapshot();
/// println!("{current_memory}")
/// ```
pub fn snapshot() -> Snapshot {
    Snapshot {
        active_memory: active_memory(),
        cache_memory: cache_memory(),
        peak_memory: peak_memory(),
    }
}

/// Get the free cache limit.
///
/// If using more than the given limit, free memory will be reclaimed
/// from the cache on the next allocation.
/// The cache limit defaults to the memory limit.
pub fn cache_limit() -> usize {
    if let Some(limit) = unsafe { CACHE_LIMIT } {
        return limit;
    }

    // sets the cache limit to a reasonable value to read, then set it back
    let current = unsafe { mlx_sys::mlx_metal_set_cache_limit(cache_memory()) };
    unsafe {
        mlx_sys::mlx_metal_set_cache_limit(current);
    }
    unsafe { CACHE_LIMIT = Some(current) }

    current
}

/// Set the free cache limit.
///
/// If using more than the given limit, free memory will be reclaimed
/// from the cache on the next allocation. To disable the cache,
/// set the limit to 0.
///
/// The cache limit defaults to the memory limit.
pub fn set_cache_limit(limit: usize) {
    unsafe { CACHE_LIMIT = Some(limit) }
    unsafe {
        mlx_sys::mlx_metal_set_cache_limit(limit);
    }
}

/// Get the memory limit.
///
/// Calls to malloc will wait on scheduled tasks if the limit is exceeded. The
/// memory limit defaults to 1.5 times the maximum recommended working set
/// size reported by the device.
///
/// See also: [set_memory_limit]
pub fn memory_limit() -> usize {
    if let Some(limit) = unsafe { MEMORY_LIMIT } {
        return limit;
    }

    // sets the memory limit to a reasonable value to read, then set it back
    let current =
        unsafe { mlx_sys::mlx_metal_set_memory_limit(active_memory(), RELAXED_MEMORY_LIMIT) };
    unsafe {
        mlx_sys::mlx_metal_set_memory_limit(current, RELAXED_MEMORY_LIMIT);
    }

    current
}

/// Set the memory limit.
///
/// Calls to malloc will wait on scheduled tasks if the limit is exceeded.  If
/// there are no more scheduled tasks an error will be raised if `relaxed`
/// is false or memory will be allocated (including the potential for
/// swap) if `relaxed` is true.
///
/// The memory limit defaults to 1.5 times the maximum recommended working set
/// size reported by the device ([recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/2369280-recommendedmaxworkingsetsize))
pub fn set_memory_limit(limit: usize, relaxed: bool) {
    unsafe { MEMORY_LIMIT = Some(limit) }
    unsafe { RELAXED_MEMORY_LIMIT = relaxed }
    unsafe {
        mlx_sys::mlx_metal_set_memory_limit(limit, relaxed);
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu;

    #[test]
    fn test_active_memory() {
        let _active_memory = gpu::active_memory();
    }

    #[test]
    fn test_cache_memory() {
        let _cache_memory = gpu::cache_memory();
    }

    #[test]
    fn test_peak_memory() {
        let _peak_memory = gpu::peak_memory();
    }

    #[test]
    fn test_cache_limit() {
        let _cache_limit = gpu::cache_limit();
    }

    // TODO: Figure an appropriate value to test
    // #[test]
    // fn test_set_cache_limit() {
    //     let cache_limit = 4096;
    //     gpu::set_cache_limit(cache_limit);
    //     println!("cache_limit: {}", gpu::cache_limit());
    //     assert_eq!(gpu::cache_limit(), cache_limit);
    // }

    #[test]
    fn test_memory_limit() {
        let _memory_limit = gpu::memory_limit();
    }

    #[test]
    fn test_set_memory_limit() {
        let memory_limit = 1024;
        gpu::set_memory_limit(memory_limit, true);
        assert_eq!(gpu::memory_limit(), memory_limit);
    }

    #[test]
    fn test_snapshot() {
        let start_memory = gpu::snapshot();
        // TODO: Use Array methods to allocate memory in GPU
        let end_memory = gpu::snapshot();

        let delta = start_memory.delta(&end_memory);
        println!("{}", delta);
    }
}
