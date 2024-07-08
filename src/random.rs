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

/// Generate uniformly distributed random numbers with a given interval (`lower_bound:` and `upper_bound:`).
///
/// The values are sampled uniformly in the range.  An optional shape can be used to broadcast into
/// a larger array.  An optional `key` can be specified to control the PRNG.
///
/// ```rust
/// let key = mlx_rs::random::key(0);
///
/// // create an array of shape `[50]` type Float values in the range [0, 10)
/// let array = mlx_rs::random::uniform::<_, f32>(0, 10, &[50], &key);
///
/// // same, but in range [0.5, 1)
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
/// will be of the given `T`. `T` must be a floating point type.
///
/// ```rust
/// let key = mlx_rs::random::key(0);
///
/// // generate a single f32 with normal distribution
/// let value = mlx_rs::random::normal::<f32>(None, None, None, &key).unwrap().item::<f32>();
///
/// // generate an array of f32 with normal distribution in shape [10, 5]
/// let array = mlx_rs::random::normal::<f32>(&[10, 5], None, None, &key);
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

/// Generate jointly-normal random samples given a mean and covariance.
///
/// The matrix `covariance` must be positive semi-definite. The behavior is
/// undefined if it is not.  The only supported output type is f32.
///
/// # Params
/// - mean: array of shape `[..., n]`, the mean of the distribution.
/// - covariance: array  of shape `[..., n, n]`, the covariance
/// matrix of the distribution. The batch shape `...` must be
/// broadcast-compatible with that of `mean`.
/// - shape: The output shape must be
/// broadcast-compatible with `mean.shape.dropLast()` and `covariance.shape.dropLast(2)`.
/// If empty, the result shape is determined by broadcasting the batch
/// shapes of `mean` and `covariance`.
/// - key: PRNG key
#[default_device]
pub fn multivariate_normal_device<'a, T: ArrayElement>(
    mean: &Array,
    covariance: &Array,
    shape: impl Into<Option<&'a [i32]>>,
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
            mlx_sys::mlx_random_multivariate_normal(
                mean.as_ptr(),
                covariance.as_ptr(),
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

/// Generate random integers from the given interval (`lower_bound:` and `upper_bound:`).
///
/// The values are sampled with equal probability from the integers in
/// half-open interval `[lb, ub)`. The lower and upper bound can be
/// scalars or arrays and must be roadcastable to `shape`.
///
/// ```rust
/// use mlx_rs::{array, random};
///
/// let key = random::key(0);
///
/// // generate an array of Int values, one in the range 0 ..< 10
/// // and one in the range 10 ..< 100
/// let array = random::randint::<_, i32>(array![0, 10], array![10, 100], None, &key);
/// ```
#[default_device]
pub fn randint_device<'a, E: Into<Array>, T: ArrayElement>(
    lower_bound: E,
    upper_bound: E,
    shape: impl Into<Option<&'a [i32]>>,
    key: impl Into<Option<&'a Array>>,
    stream: StreamOrDevice,
) -> Result<Array, Exception> {
    let lb: Array = lower_bound.into();
    let ub: Array = upper_bound.into();
    let shape = shape.into().unwrap_or(lb.shape());

    let key = key.into().map_or_else(
        || {
            let mut state = state().lock().unwrap();
            state.next()
        },
        |key| key.clone(),
    );

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_random_randint(
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

/// Generate Bernoulli random values with a given `p` value.
///
/// The values are sampled from the bernoulli distribution with parameter
/// `p`. The parameter `p` must have a floating point type and
/// must be broadcastable to `shape`.
///
/// ```rust
/// use mlx_rs::{array, Array, random};
///
/// let key = random::key(0);
///
/// // generate a single random Bool with p = 0.8
/// let p: Array = 0.8.into();
/// let value = random::bernoulli(&p, None, &key);
///
/// // generate an array of shape [50, 2] of random Bool with p = 0.8
/// let array = random::bernoulli(&p, &[50, 2], &key);
///
/// // generate an array of [3] Bool with the given p values
/// let array = random::bernoulli(&array![0.1, 0.5, 0.8], None, &key);
/// ```
#[default_device]
pub fn bernoulli_device<'a>(
    p: impl Into<Option<&'a Array>>,
    shape: impl Into<Option<&'a [i32]>>,
    key: impl Into<Option<&'a Array>>,
    stream: StreamOrDevice,
) -> Result<Array, Exception> {
    let default_array = Array::from_float(0.5);
    let p = p.into().unwrap_or(&default_array);

    let shape = shape.into().unwrap_or(p.shape());

    let key = key.into().map_or_else(
        || {
            let mut state = state().lock().unwrap();
            state.next()
        },
        |key| key.clone(),
    );

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_random_bernoulli(
                p.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                key.as_ptr(),
                stream.as_ptr(),
            )
        };
        Ok(Array::from_ptr(c_array))
    }
}

/// Generate values from a truncated normal distribution between `low` and `high`.
///
/// The values are sampled from the truncated normal distribution
/// on the domain `(lower, upper)`. The bounds `lower` and `upper`
/// can be scalars or arrays and must be broadcastable to `shape`.
///
/// ```rust
/// use mlx_rs::{array, random};
///
/// let key = random::key(0);
///
/// // generate an array of two Float values, one in the range 0 ..< 10
/// // and one in the range 10 ..< 100
/// let value = random::truncated_normal::<_, f32>(array![0, 10], array![10, 100], None, &key);
/// ```
#[default_device]
pub fn truncated_normal_device<'a, E: Into<Array>, T: ArrayElement>(
    lower_bound: E,
    upper_bound: E,
    shape: impl Into<Option<&'a [i32]>>,
    key: impl Into<Option<&'a Array>>,
    stream: StreamOrDevice,
) -> Result<Array, Exception> {
    let lb: Array = lower_bound.into();
    let ub: Array = upper_bound.into();
    let shape = shape.into().unwrap_or(lb.shape());

    let key = key.into().map_or_else(
        || {
            let mut state = state().lock().unwrap();
            state.next()
        },
        |key| key.clone(),
    );

    unsafe {
        let c_array = try_catch_c_ptr_expr! {
            mlx_sys::mlx_random_truncated_normal(
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

/// Sample from the standard Gumbel distribution.
///
/// The values are sampled from a standard Gumbel distribution
/// which CDF `exp(-exp(-x))`.
///
/// ```rust
/// let key = mlx_rs::random::key(0);
///
/// // generate a single Float with Gumbel distribution
/// let value = mlx_rs::random::gumbel::<f32>(None, &key).unwrap().item::<f32>();
///
/// // generate an array of Float with Gumbel distribution in shape [10, 5]
/// let array = mlx_rs::random::gumbel::<f32>(&[10, 5][..], &key);
/// ```
#[default_device]
pub fn gumbel_device<'a, T: ArrayElement>(
    shape: impl Into<Option<&'a [i32]>>,
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
            mlx_sys::mlx_random_gumbel(
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

/// Sample from a categorical distribution.
///
/// The values are sampled from the categorical distribution specified by
/// the unnormalized values in `logits`.   If the `shape` is not specified
/// the result shape will be the same shape as `logits` with the `axis`
/// dimension removed.
///
/// ```rust
/// let key = mlx_rs::random::key(0);
///
/// let logits = mlx_rs::Array::zeros::<u32>(&[5, 20]).unwrap();
///
/// // produces Array of u32 shape &[5]
/// let result = mlx_rs::random::categorical(&logits, None, None, None, &key);
/// ```
///
/// - Parameters:
///     - logits: The *unnormalized* categorical distribution(s).
#[default_device]
pub fn categorical_device<'a>(
    logits: &Array,
    axis: impl Into<Option<i32>>,
    shape: impl Into<Option<&'a [i32]>>,
    num_samples: impl Into<Option<i32>>,
    key: impl Into<Option<&'a Array>>,
    stream: StreamOrDevice,
) -> Result<Array, Exception> {
    let axis = axis.into().unwrap_or(-1);

    let key = key.into().map_or_else(
        || {
            let mut state = state().lock().unwrap();
            state.next()
        },
        |key| key.clone(),
    );

    let shape = shape.into();
    let num_samples = num_samples.into();

    // if we have both shape and num_samples, we need to error out
    if shape.is_some() && num_samples.is_some() {
        let what = "[categorical] Cannot specify both shape and num_samples";
        let c_string = std::ffi::CString::new(what).expect("CString::new failed");

        return Err(Exception { what: c_string });
    }

    if let Some(shape) = shape {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_random_categorical_shape(
                    logits.as_ptr(),
                    axis,
                    shape.as_ptr(),
                    shape.len(),
                    key.as_ptr(),
                    stream.as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    } else if let Some(num_samples) = num_samples {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_random_categorical_num_samples(
                    logits.as_ptr(),
                    axis,
                    num_samples,
                    key.as_ptr(),
                    stream.as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    } else {
        unsafe {
            let c_array = try_catch_c_ptr_expr! {
                mlx_sys::mlx_random_categorical(
                    logits.as_ptr(),
                    axis,
                    key.as_ptr(),
                    stream.as_ptr(),
                )
            };
            Ok(Array::from_ptr(c_array))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array, assert_array_eq};
    use float_eq::float_eq;

    #[test]
    fn test_global_rng() {
        seed(3);
        let a = uniform::<_, f32>(0, 1, &[], None).unwrap();
        let b = uniform::<_, f32>(0, 1, &[], None).unwrap();

        seed(3);
        let x = uniform::<_, f32>(0, 1, &[], None).unwrap();
        let y = uniform::<_, f32>(0, 1, &[], None).unwrap();

        assert_array_eq!(a, x, 0.01);
        assert_array_eq!(b, y, 0.01);
    }

    #[test]
    fn test_key() {
        let k1 = key(0);
        let k2 = key(0);
        assert!(k1 == k2);

        let k2 = key(1);
        assert!(k1 != k2);
    }

    #[test]
    fn test_split() {
        let key = key(0);

        let (k1, k2) = split(&key);
        assert!(k1 != k2);

        let (r1, r2) = split(&key);
        assert!(r1 == k1);
        assert!(r2 == k2);
    }

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

    #[test]
    fn test_multivariate_normal() {
        let key = key(0);
        let mean = Array::from_slice(&[0.0, 0.0], &[2]);
        let covariance = Array::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let a = multivariate_normal::<f32>(&mean, &covariance, &[3][..], &key).unwrap();
        assert!(a.shape() == [3, 2]);
    }

    #[test]
    fn test_randint_single() {
        let key = key(0);
        let mut value = randint::<_, i32>(0, 100, None, Some(&key)).unwrap();
        assert_eq!(value.item::<i32>(), 41);
    }

    #[test]
    fn test_randint_multiple() {
        let key = key(0);
        let value = randint::<_, i32>(array![0, 10], array![10, 100], None, Some(&key)).unwrap();
        let expected = Array::from_slice(&[2, 82], &[2]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_randint_non_int() {
        let key = key(0);
        let value = randint::<_, f32>(array![0, 10], array![10, 100], None, Some(&key));
        assert!(value.is_err());
    }

    #[test]
    fn test_bernoulli_single() {
        let key = key(0);
        let mut value = bernoulli(None, None, &key).unwrap();
        assert_eq!(value.item::<bool>(), true);
    }

    #[test]
    fn test_bernoulli_multiple() {
        let key = key(0);
        let value = bernoulli(None, &[4][..], &key).unwrap();
        let expected = Array::from_slice(&[false, true, false, true], &[4]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_bernoulli_p() {
        let key = key(0);
        let p: Array = 0.8.into();
        let value = bernoulli(&p, &[4][..], &key).unwrap();
        let expected = Array::from_slice(&[false, true, true, true], &[4]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_bernoulli_p_array() {
        let key = key(0);
        let value = bernoulli(&array![0.1, 0.5, 0.8], None, &key).unwrap();
        let expected = Array::from_slice(&[false, true, true], &[3]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_truncated_normal_single() {
        let key = key(0);
        let value = truncated_normal::<_, f32>(0, 10, None, &key).unwrap();
        assert_array_eq!(value, Array::from_float(0.55), 0.01);
    }

    #[test]
    fn test_truncated_normal_multiple() {
        let key = key(0);
        let value = truncated_normal::<_, f32>(0.0, 0.5, &[3][..], &key).unwrap();
        let expected = Array::from_slice(&[0.48, 0.15, 0.30], &[3]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_truncated_normal_multiple_array() {
        let key = key(0);
        let value =
            truncated_normal::<_, f32>(array![0.0, 0.5], array![0.5, 1.0], None, &key).unwrap();
        let expected = Array::from_slice(&[0.10, 0.88], &[2]);

        assert_array_eq!(value, expected, 0.01);
    }

    #[test]
    fn test_gumbel() {
        let key = key(0);
        let value = gumbel::<f32>(None, &key).unwrap();
        assert_array_eq!(value, Array::from_float(0.13), 0.01);
    }

    #[test]
    fn test_logits() {
        let key = key(0);
        let logits = Array::zeros::<u32>(&[5, 20]).unwrap();
        let result = categorical(&logits, None, None, None, &key).unwrap();

        assert!(result.shape() == [5]);

        let expected = Array::from_slice(&[1, 1, 17, 17, 17], &[5]);
        assert_array_eq!(result, expected, 0.01);
    }

    #[test]
    fn test_logits_count() {
        let key = key(0);
        let logits = Array::zeros::<u32>(&[5, 20]).unwrap();
        let result = categorical(&logits, None, None, 2, &key).unwrap();

        assert!(result.shape() == [5, 2]);

        let expected = Array::from_slice(&[16, 3, 14, 10, 17, 7, 6, 8, 12, 8], &[5, 2]);
        assert_array_eq!(result, expected, 0.01);
    }
}
