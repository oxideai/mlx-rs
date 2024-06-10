use smallvec::SmallVec;

use crate::{utils::resolve_index_unchecked, Array};

#[inline]
pub(super) fn resolve_size_and_axis_unchecked(
    a: &Array,
    n: Option<i32>,
    axis: Option<i32>,
) -> (i32, i32) {
    let axis = axis.unwrap_or(-1);
    let n = n.unwrap_or_else(|| {
        let axis_index = resolve_index_unchecked(axis, a.ndim());
        a.shape()[axis_index]
    });
    (n, axis)
}

// Use Cow or SmallVec?
#[inline]
pub(super) fn resolve_sizes_and_axes_unchecked<'a>(
    a: &Array,
    s: Option<&'a [i32]>,
    axes: Option<&'a [i32]>,
) -> (SmallVec<[i32; 4]>, SmallVec<[i32; 4]>) {
    match (s, axes) {
        (Some(s), Some(axes)) => {
            let valid_s = SmallVec::<[i32; 4]>::from_slice(s);
            let valid_axes = SmallVec::<[i32; 4]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (Some(s), None) => {
            let valid_s = SmallVec::<[i32; 4]>::from_slice(s);
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
        (None, Some(axes)) => {
            let valid_s = axes
                .iter()
                .map(|&axis| {
                    let axis_index = resolve_index_unchecked(axis, a.ndim());
                    a.shape()[axis_index]
                })
                .collect();
            let valid_axes = SmallVec::<[i32; 4]>::from_slice(axes);
            (valid_s, valid_axes)
        }
        (None, None) => {
            let valid_s: SmallVec<[i32; 4]> = (0..a.ndim()).map(|axis| a.shape()[axis]).collect();
            let valid_axes = (-(valid_s.len() as i32)..0).collect();
            (valid_s, valid_axes)
        }
    }
}
