use autocxx::prelude::*;

include_cpp! {
    #include "mlx/mlx.h"
    #include "ext.h"
    // #include "extras.h" // TODO: remove this later
    // TODO: what safety option should be used here?
    safety!(unsafe)

    generate!("ext::hello")
    generate_ns!("ext::array")

    // mlx/mlx/allocator.h
    generate!("mlx::core::allocator::Buffer")
    
    // mlx/mlx/dtype.h
    generate!("mlx::core::Dtype")
    generate!("mlx::core::is_available")
    generate!("mlx::core::promote_types")
    generate!("mlx::core::kindof")
    generate!("mlx::core::is_unsigned")
    generate!("mlx::core::is_floating_point")
    generate!("mlx::core::is_complex")
    generate!("mlx::core::is_integral")
    generate!("mlx::core::dtype_to_array_protocol")
    generate!("mlx::core::dtype_from_array_protocol")
    
    // mlx/mlx/backend/metal/metal.h
    generate_ns!("mlx::core::metal")
    
    // mlx/mlx/device.h
    generate!("mlx::core::Device")
    generate!("mlx::core::set_default_device")
    
    // mlx/mlx/fft.h
    generate_ns!("mlx::core::fft")

    // mlx/mlx/ops.h
    // TODO:

    // mlx/mlx/random.h
    generate_ns!("mlx::core::random")

    // mlx/mlx/stream.h
    generate!("mlx::core::Stream")
    generate!("mlx::core::default_stream")
    generate!("mlx::core::set_default_stream")
    generate!("mlx::core::new_stream")

    // mlx/mlx/transforms.h
    // TODO:
    // generate!("mlx::core::simplify")

    // mlx/mlx/utils.h
    generate!("mlx::core::result_type")
    generate!("mlx::core::broadcast_shapes")
    generate!("mlx::core::is_same_shape")
    generate!("mlx::core::normalize_axis")

    generate!("mlx::core::deleter_t") // TODO: somehow this is fine

    // TODO: Unsupported
    // generate!("mlx::core::TypeToDtype") // TODO: template specialization, see concrete!
    // generate!("mlx::core::default_device") // bindings cannot be generated
    // generate!("mlx::core::array") // TODO: array is not supported by autocxx

}

fn main() {
    let hello = ffi::ext::hello();
    println!("{}", hello);

    let key = ffi::mlx::core::random::key(1);
}