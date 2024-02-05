use cxx::{CxxVector, UniquePtr};

use crate::Optional;

type OptionalAxis = Optional<UniquePtr<CxxVector<i32>>>;

unsafe impl cxx::ExternType for OptionalAxis {
    type Id = cxx::type_id!("mlx_cxx::OptionalAxis");

    type Kind = cxx::kind::Opaque;
}

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx-cxx/linalg.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        type OptionalAxis = crate::linalg::OptionalAxis;

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::StreamOrDevice;

        #[namespace = "mlx_cxx"]
        fn norm_ord(
            a: &array,
            ord: f64,
            axis: &OptionalAxis,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn norm_ord_axis(
            a: &array,
            ord: f64,
            axis: i32,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn norm_str_ord(
            a: &array,
            ord: &CxxString,
            axis: &OptionalAxis,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn norm_str_ord_axis(
            a: &array,
            ord: &CxxString,
            axis: i32,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn norm(
            a: &array,
            axis: &OptionalAxis,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;

        #[namespace = "mlx_cxx"]
        fn norm_axis(
            a: &array,
            axis: i32,
            keepdims: bool,
            s: StreamOrDevice,
        ) -> Result<UniquePtr<array>>;
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::cxx_vec;

//     #[test]
//     fn test_norm_optional_axis() {
//         let slice = &[1.0, 2.0, 3.0];
//         let shape = cxx_vec![3];
//         let a = crate::array::ffi::array_from_slice_float32(slice, &shape);
//         let norm =
//             super::ffi::norm(&a, &super::Optional::None, false, Default::default());

//         assert_eq!(norm.size(), 1);
//     }
// }
