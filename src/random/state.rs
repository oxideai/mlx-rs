use crate::random::key;
use crate::Array;

struct RustKeySequence {
    state: Array,
}

impl RustKeySequence {
    fn seed(&mut self, seed: u64) {
        self.state = key(seed)
    }

    fn next(&mut self) -> Array {
        let state = self.state.clone();
        let (state, key) = state.split_at(1);
        self.state = key;
        state
    }
}

impl Default for RustKeySequence {
    fn default() -> Self {
        let now = std::time::SystemTime::now();
        let now = now.elapsed().unwrap().as_secs();
        Self { state: key(now) }
    }
}
