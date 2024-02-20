#[repr(C, u8)]
pub enum StreamOrDevice {
    Default,
    Stream(ffi::Stream),
    Device(ffi::Device),
}

impl Default for StreamOrDevice {
    fn default() -> Self {
        Self::Default
    }
}

unsafe impl cxx::ExternType for StreamOrDevice {
    type Id = cxx::type_id!("mlx_cxx::StreamOrDevice");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/utils.h");
        include!("mlx-cxx/utils.hpp");
        include!("mlx/stream.h");
        include!("mlx/device.h");
        include!("mlx-cxx/mlx_cxx.hpp");

        #[namespace = "mlx::core"]
        type Stream = crate::stream::ffi::Stream;

        #[namespace = "mlx::core"]
        type Device = crate::device::ffi::Device;

        #[namespace = "mlx_cxx"]
        type StreamOrDevice = crate::utils::StreamOrDevice;

        #[namespace = "mlx::core"]
        type Dtype = crate::dtype::ffi::Dtype;

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        #[namespace = "mlx_cxx"]
        fn result_type(arrays: &[UniquePtr<array>]) -> Dtype;

        #[namespace = "mlx_cxx"]
        fn broadcast_shapes(
            s1: &CxxVector<i32>,
            s2: &CxxVector<i32>,
        ) -> Result<UniquePtr<CxxVector<i32>>>;

        #[namespace = "mlx_cxx"]
        fn is_same_shape(arrays: &[UniquePtr<array>]) -> bool;

        #[namespace = "mlx::core"]
        fn normalize_axis(axis: i32, ndim: i32) -> Result<i32>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "push_opaque"]
        fn push_back_array(vec: Pin<&mut CxxVector<array>>, array: UniquePtr<array>);

        #[namespace = "mlx_cxx"]
        #[cxx_name = "pop_opaque"]
        fn pop_back_array(vec: Pin<&mut CxxVector<array>>) -> UniquePtr<array>;

        #[namespace = "mlx_cxx"]
        #[cxx_name = "std_vec_from_slice"]
        fn new_cxx_vec_array_from_slice(slice: &[UniquePtr<array>]) -> UniquePtr<CxxVector<array>>;
    }
}
