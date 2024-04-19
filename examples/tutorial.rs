use mlx::{Array, Dtype};
use std::ops::Add;

fn scalar_basics() {
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

fn array_basics() {
    // make a multidimensional array.
    let x: Array = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // mlx is row-major by default so the first row of this array
    // is [1.0, 2.0] and the second row is [3.0, 4.0]

    // Make an array of shape {2, 2} filled with ones:
    let y = Array::ones::<f32>(&[2, 2]);

    // Pointwise add x and y:
    let mut z = x.add(&y);

    // Same thing:
    z = &x + &y;

    // mlx is lazy by default. At this point `z` only
    // has a shape and a type but no actual data:
    assert_eq!(z.dtype(), Dtype::Float32);
    assert_eq!(z.shape(), vec![2, 2]);

    // To actually run the computation you must evaluate `z`.
    // Under the hood, mlx records operations in a graph.
    // The variable `z` is a node in the graph which points to its operation
    // and inputs. When `eval` is called on an array (or arrays), the array and
    // all of its dependencies are recursively evaluated to produce the result.
    // Once an array is evaluated, it has data and is detached from its inputs.
    z.eval();

    // Of course the array can still be an input to other operations. You can even
    // call eval on the array again, this will just be a no-op:
    z.eval(); // no-op

    // Some functions or methods on arrays implicitly evaluate them. For example
    // accessing a value in an array or printing the array implicitly evaluate it:
    z = Array::ones::<f32>(&[1]);
    z.item::<f32>(); // implicit evaluation

    z = Array::ones::<f32>(&[2, 2]);
    println!("{}", z); // implicit evaluation
}

fn main() {
    scalar_basics();
    array_basics();
}
