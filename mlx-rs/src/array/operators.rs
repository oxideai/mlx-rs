use crate::{utils::ScalarOrArray, Array, StreamOrDevice};
use num_traits::Pow;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $c_method:ident) => {
        impl<'a, T> $trait<T> for Array
        where
            T: ScalarOrArray<'a>,
        {
            type Output = Array;

            fn $method(self, rhs: T) -> Self::Output {
                paste::paste! {
                    self.[<$c_method _device>](rhs.into_owned_or_ref_array(), StreamOrDevice::default()).unwrap()
                }
            }
        }

        impl<'a, 't: 'a, T> $trait<T> for &'a Array
        where
            T: ScalarOrArray<'t>,
        {
            type Output = Array;

            fn $method(self, rhs: T) -> Self::Output {
                paste::paste! {
                    self.[<$c_method _device>](rhs.into_owned_or_ref_array(), StreamOrDevice::default()).unwrap()
                }
            }
        }
    };
}

macro_rules! impl_binary_op_assign {
    ($trait:ident, $method:ident, $c_method:ident) => {
        impl<T: Into<Array>> $trait<T> for Array {
            fn $method(&mut self, rhs: T) {
                let new_array = paste::paste! {
                    self.[<$c_method _device>](&rhs.into(), StreamOrDevice::default()).unwrap()
                };
                *self = new_array;
            }
        }

        impl $trait<&Array> for Array {
            fn $method(&mut self, rhs: &Self) {
                let new_array = paste::paste! {
                    self.[<$c_method _device>](rhs, StreamOrDevice::default()).unwrap()
                };
                *self = new_array;
            }
        }
    };
}

impl_binary_op!(Add, add, add);
impl_binary_op_assign!(AddAssign, add_assign, add);
impl_binary_op!(Sub, sub, subtract);
impl_binary_op_assign!(SubAssign, sub_assign, subtract);
impl_binary_op!(Mul, mul, multiply);
impl_binary_op_assign!(MulAssign, mul_assign, multiply);
impl_binary_op!(Div, div, divide);
impl_binary_op_assign!(DivAssign, div_assign, divide);
impl_binary_op!(Rem, rem, remainder);
impl_binary_op_assign!(RemAssign, rem_assign, remainder);
impl_binary_op!(Pow, pow, power);

impl Neg for &Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        self.negative_device(StreamOrDevice::default()).unwrap()
    }
}
impl Neg for Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        self.negative_device(StreamOrDevice::default()).unwrap()
    }
}

impl Not for &Array {
    type Output = Array;
    fn not(self) -> Self::Output {
        self.logical_not_device(StreamOrDevice::default())
    }
}
impl Not for Array {
    type Output = Array;
    fn not(self) -> Self::Output {
        self.logical_not_device(StreamOrDevice::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_add_assign() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
        a += &b;

        assert_eq!(a.as_slice::<f32>(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_assign() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
        a -= &b;

        assert_eq!(a.as_slice::<f32>(), &[-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_mul_assign() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
        a *= &b;

        assert_eq!(a.as_slice::<f32>(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_div_assign() {
        let mut a = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0, 5.0, 6.0], &[3]);
        a /= &b;

        assert_eq!(a.as_slice::<f32>(), &[0.25, 0.4, 0.5]);
    }
}
