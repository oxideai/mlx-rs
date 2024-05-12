use crate::prelude::IndexOp;
use crate::{Array, ArrayElement, StreamOrDevice};
use mlx_macros::default_device;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

/// Returns a seeded random number generator using entropy.
#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    StdRng::from_entropy()
}

/// Seed the random number generator.
pub fn seed(seed: u64) {
    let rng = StdRng::seed_from_u64(seed);
    let mut seed = SEED.lock().unwrap();
    *seed = Some(rng);
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
/// let key = mlx::random::key(0);
///
/// // create an array of shape `[50]` type Float values in the range `0 ..< 10`
/// let array = mlx::random::uniform::<_, f32>(0, 10, &[50], &key);
///
/// // same, but in range `0.5 ..< 1`
/// let array = mlx::random::uniform::<_, f32>(0.5f32, 1f32, &[50], &key);
/// ```
#[default_device]
pub fn uniform_device<'a, E: Into<Array>, T: ArrayElement>(
    lower_bound: E,
    upper_bound: E,
    shape: &[i32],
    key: impl Into<Option<&'a Array>>,
    stream: StreamOrDevice,
) -> Array {
    let lb: Array = lower_bound.into();
    let ub: Array = upper_bound.into();

    let mut seed = SEED.lock().unwrap();
    let mut rng = match seed.as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => get_seeded_rng(),
    };

    let key = key.into().map_or_else(
        || {
            let key: i32 = rng.gen();
            Array::from_int(key)
        },
        |key| key.clone(),
    );

    let array = unsafe {
        Array::from_ptr(mlx_sys::mlx_random_uniform(
            lb.as_ptr(),
            ub.as_ptr(),
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            key.as_ptr(),
            stream.as_ptr(),
        ))
    };

    *seed = Some(rng);
    array
}

/// Generate normally distributed random numbers.
///
/// Generate an array of random numbers using the optional shape. The result
/// will be of the given `T`.
///
/// ```rust
/// let key = mlx::random::key(0);
///
/// // generate a single Float with normal distribution
/// let value = mlx::random::normal::<f32>(None, None, None, &key).item::<f32>();
///
/// // generate an array of Float with normal distribution in shape [10, 5]
/// let array = mlx::random::normal::<f32>(&[10, 5][..], None, None, &key);
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
) -> Array {
    let shape = shape.into().unwrap_or(&[]);
    let mut seed = SEED.lock().unwrap();
    let mut rng = match seed.as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => get_seeded_rng(),
    };

    let key = key.into().map_or_else(
        || {
            let key: i32 = rng.gen();
            Array::from_int(key)
        },
        |key| key.clone(),
    );

    let array = unsafe {
        Array::from_ptr(mlx_sys::mlx_random_normal(
            shape.as_ptr(),
            shape.len(),
            T::DTYPE.into(),
            loc.into().unwrap_or(0.0),
            scale.into().unwrap_or(1.0),
            key.as_ptr(),
            stream.as_ptr(),
        ))
    };

    *seed = Some(rng);
    array
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_array_eq;
    use float_eq::float_eq;

    #[test]
    fn test_uniform_single() {
        let key = key(0);
        let mut value = uniform::<_, f32>(0, 10, &[], Some(&key));
        float_eq!(value.item::<f32>(), 4.18, abs <= 0.01);
    }

    #[test]
    fn test_uniform_multiple() {
        let key = key(0);
        let value = uniform::<_, f32>(0, 10, &[3], Some(&key));
        let expected = Array::from_slice(&[9.65, 3.14, 6.33], &[3]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_uniform_multiple_array() {
        let key = key(0);
        let value = uniform::<_, f32>(&[0, 10][..], &[10, 100][..], &[2], Some(&key));
        let expected = Array::from_slice(&[2.16, 82.37], &[2]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_normal() {
        let key = key(0);
        let mut value = normal::<f32>(None, None, None, &key);
        float_eq!(value.item::<f32>(), -0.20, abs <= 0.01);
    }
}
