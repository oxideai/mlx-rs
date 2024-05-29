use crate::{Array, StreamOrDevice};
use num_traits::Pow;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident) => {
        impl<T: Into<Array>> $trait<T> for Array {
            type Output = Array;
            fn $method(self, rhs: T) -> Self::Output {
                paste::paste! {
                    self.[<$method _device>](&rhs.into(), StreamOrDevice::default())
                }
            }
        }

        impl<'a, T: Into<Array>> $trait<T> for &'a Array {
            type Output = Array;
            fn $method(self, rhs: T) -> Self::Output {
                paste::paste! {
                    self.[<$method _device>](&rhs.into(), StreamOrDevice::default()).unwrap()
                }
            }
        }

        impl<'a> $trait for &'a Array {
            type Output = Array;
            fn $method(self, rhs: Self) -> Self::Output {
                paste::paste! {
                    self.[<$method _device>](rhs, StreamOrDevice::default()).unwrap()
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

impl_binary_op!(Add, add);
impl_binary_op_assign!(AddAssign, add_assign, add);
impl_binary_op!(Sub, sub);
impl_binary_op_assign!(SubAssign, sub_assign, sub);
impl_binary_op!(Mul, mul);
impl_binary_op_assign!(MulAssign, mul_assign, mul);
impl_binary_op!(Div, div);
impl_binary_op_assign!(DivAssign, div_assign, div);
impl_binary_op!(Rem, rem);

impl<'a> Pow<&'a Array> for &'a Array {
    type Output = Array;
    fn pow(self, rhs: &'a Array) -> Self::Output {
        self.pow_device(rhs, StreamOrDevice::default()).unwrap()
    }
}

impl<'a, T: Into<Array>> Pow<T> for &'a Array {
    type Output = Array;
    fn pow(self, rhs: T) -> Self::Output {
        self.pow_device(&rhs.into(), StreamOrDevice::default())
            .unwrap()
    }
}

impl<'a> Neg for &'a Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        self.logical_not()
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
