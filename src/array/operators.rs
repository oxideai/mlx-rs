use crate::{Array, StreamOrDevice};
use num_traits::Pow;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident) => {
        impl<'a, T: Into<Array>> $trait<T> for &'a Array {
            type Output = Array;
            fn $method(self, rhs: T) -> Self::Output {
                paste::paste! {
                    self.[<$method _device>](&rhs.into(), StreamOrDevice::default())
                }
            }
        }

        impl<'a> $trait for &'a Array {
            type Output = Array;
            fn $method(self, rhs: Self) -> Self::Output {
                paste::paste! {
                    self.[<$method _device>](rhs, StreamOrDevice::default())
                }
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
impl_binary_op!(Rem, rem);

impl<'a> Pow<&'a Array> for &'a Array {
    type Output = Array;
    fn pow(self, rhs: &'a Array) -> Self::Output {
        self.pow_device(rhs, StreamOrDevice::default())
    }
}

impl<'a, T: Into<Array>> Pow<T> for &'a Array {
    type Output = Array;
    fn pow(self, rhs: T) -> Self::Output {
        self.pow_device(&rhs.into(), StreamOrDevice::default())
    }
}

impl<'a> Neg for &'a Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        self.logical_not()
    }
}
