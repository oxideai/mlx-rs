use crate::prelude::IndexOp;
use crate::{error::Exception, Array, ArrayElement, StreamOrDevice};
use mach_sys::mach_time;
use mlx_macros::default_device;
use std::sync::{Mutex, OnceLock};

struct RandomState {
    state: Array,
}

impl RandomState {
    fn new() -> Self {
        let now = unsafe { mach_time::mach_approximate_time() };
        Self { state: key(now) }
    }

    fn next(&mut self) -> Array {
        let next = split(&self.state);
        self.state = next.0;
        next.1
    }

    fn seed(&mut self, seed: u64) {
        self.state = key(seed);
    }
}

fn state() -> &'static Mutex<RandomState> {
    static STATE: OnceLock<Mutex<RandomState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(RandomState::new()))
}

/// Seed the random number generator.
pub fn seed(seed: u64) {
    let mut state = state().lock().unwrap();
    state.seed(seed);
}

/// Get a PRNG key from a seed.
///
/// Return a value that can be used as a PRNG key.  All ``random::*``
/// functions take an optional key -- this will let you control the
/// random number generation.
pub fn key(seed: u64) -> Array {
    unsafe { Array::from_ptr(mlx_sys::mlx_random_key(seed)) }
}

/// Split a PRNG key into two keys and return a tuple.
#[default_device]
pub fn split_device(key: &Array, stream: StreamOrDevice) -> (Array, Array) {
    let keys = unsafe {
        Array::from_ptr(mlx_sys::mlx_random_split_equal_parts(
            key.as_ptr(),
            2,
            stream.as_ptr(),
        ))
    };

    (keys.index(0), keys.index(1))
}

/// Generate uniformly distributed random numbers with a `RangeBounds`.
///
/// The values are sampled uniformly in the range.  An optional shape can be used to broadcast into
/// a larger array.  An optional `key` can be specified to control the PRNG.
///
/// ```rust
/// let key = mlx_rs::random::key(0);
///
/// // create an array of shape `[50]` type Float values in the range `0 ..< 10`
/// let array = mlx_rs::random::uniform::<_, f32>(0, 10, &[50], &key);
///
/// // same, but in range `0.5 ..< 1`
/// let array = mlx_rs::random::uniform::<_, f32>(0.5f32, 1f32, &[50], &key);
/// ```
#[default_device]
pub fn uniform_device<'a, E: Into<Array>, T: ArrayElement>(
    lower_bound: E,
    upper_bound: E,
    shape: &[i32],
    key: impl Into<Option<&'a Array>>,
    stream: StreamOrDevice,
) -> Result<Array, Exception> {
    let lb: Array = lower_bound.into();
    let ub: Array = upper_bound.into();

    let key = key.into().map_or_else(
        || {
            let mut state = state().lock().unwrap();
            state.next()
        },
        |key| key.clone(),
    );

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_random_uniform(
                lb.as_ptr(),
                ub.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
                key.as_ptr(),
                stream.as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

/// Generate normally distributed random numbers.
///
/// Generate an array of random numbers using the optional shape. The result
/// will be of the given `T`.
///
/// ```rust
/// let key = mlx_rs::random::key(0);
///
/// // generate a single Float with normal distribution
/// let value = mlx_rs::random::normal::<f32>(None, None, None, &key).item::<f32>();
///
/// // generate an array of Float with normal distribution in shape [10, 5]
/// let array = mlx_rs::random::normal::<f32>(&[10, 5][..], None, None, &key);
/// ```
///
/// # Params
///  - shape: shape of the output, if `None` a single value is returned
///  - loc: mean of the distribution, default is `0.0`
///  - scale: standard deviation of the distribution, default is `1.0`
///  - key: PRNG key
#[default_device]
pub fn normal_device<'a, T: ArrayElement>(
    shape: impl Into<Option<&'a [i32]>>,
    loc: impl Into<Option<f32>>,
    scale: impl Into<Option<f32>>,
    key: impl Into<Option<&'a Array>>,
    stream: StreamOrDevice,
) -> Result<Array, Exception> {
    let shape = shape.into().unwrap_or(&[]);

    let key = key.into().map_or_else(
        || {
            let mut state = state().lock().unwrap();
            state.next()
        },
        |key| key.clone(),
    );

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_random_normal(
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
                loc.into().unwrap_or(0.0),
                scale.into().unwrap_or(1.0),
                key.as_ptr(),
                stream.as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_array_eq;
    use float_eq::float_eq;

    #[test]
    fn test_uniform_no_seed() {
        let value = uniform::<_, f32>(0, 10, &[3], None).unwrap();
        assert_eq!(value.shape(), &[3]);
    }

    #[test]
    fn test_uniform_single() {
        let key = key(0);
        let mut value = uniform::<_, f32>(0, 10, &[], Some(&key)).unwrap();
        float_eq!(value.item::<f32>(), 4.18, abs <= 0.01);
    }

    #[test]
    fn test_uniform_multiple() {
        let key = key(0);
        let value = uniform::<_, f32>(0, 10, &[3], Some(&key)).unwrap();
        let expected = Array::from_slice(&[9.65, 3.14, 6.33], &[3]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_uniform_multiple_array() {
        let key = key(0);
        let value = uniform::<_, f32>(&[0, 10][..], &[10, 100][..], &[2], Some(&key)).unwrap();
        let expected = Array::from_slice(&[2.16, 82.37], &[2]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_uniform_non_float() {
        let key = key(0);
        let value = uniform::<_, i32>(&[0, 10][..], &[10, 100][..], &[2], Some(&key));
        assert!(value.is_err());
    }

    #[test]
    fn test_normal() {
        let key = key(0);
        let mut value = normal::<f32>(None, None, None, &key).unwrap();
        float_eq!(value.item::<f32>(), -0.20, abs <= 0.01);
    }

    #[test]
    fn test_normal_non_float() {
        let key = key(0);
        let value = normal::<i32>(None, None, None, &key);
        assert!(value.is_err());
    }
}
