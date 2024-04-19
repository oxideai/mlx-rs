use mlx::Array;

fn main() {
    // create a scalar array
    let x: Array = 1.0.into();

    // the datatype is .float32
    let dtype = x.dtype();
    assert_eq!(dtype, mlx::Dtype::Float32);

    // get the value
    let s = x.item::<f32>();
    assert_eq!(s, 1.0);

    // reading the value with a different type is a fatal error
    // let i = x.item::<i32>();

    // scalars have a size of 1
    let size = x.size();
    assert_eq!(size, 1);

    // scalars have 0 dimensions
    let ndim = x.ndim();
    assert_eq!(ndim, 0);

    // scalar shapes are empty arrays
    let shape = x.shape();
    assert_eq!(shape, vec![]);
}
