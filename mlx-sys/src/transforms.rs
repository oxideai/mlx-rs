//! The function pointer used in functions like `grad()` are used to
//! eventually compute the output `array` which is then used to compute the gradient.
//! So theoretically we could instead pass a rust function that is callable from C++.

pub trait Function<Args> {
    type Output;

    fn execute(&self, args: Args) -> Self::Output;
}

impl<Args, F, R> Function<Args> for F
where
    F: Fn(Args) -> R,
{
    type Output = R;

    fn execute(&self, args: Args) -> Self::Output {
        self(args)
    }
}

type DynFn = Box<dyn Function<i32, Output=i32>>;

fn execute_dyn_fn(f: &DynFn, args: i32) -> i32 {
    f.execute(args)
}

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("mlx/array.h");
        include!("mlx-cxx/transforms.hpp");

        #[namespace = "mlx::core"]
        type array = crate::array::ffi::array;

        type MultiaryFn;

        // TODO: This clearly changes internal states of the arrays. We should review if it should
        // be put behind a mut reference.
        #[namespace = "mlx_cxx"]
        fn simplify(outputs: &[UniquePtr<array>]);

        // TODO: This clearly changes internal states of the arrays. We should review if it should
        // be put behind a mut reference.
        #[namespace = "mlx_cxx"]
        fn eval(outputs: &[UniquePtr<array>]);

        // // TODO: this needs to be placed in a separate file
        // #[namespace = "mlx_cxx"]
        // fn execute_callback(f: &DynFn, args: i32) -> i32;

        #[namespace = "mlx_cxx"]
        fn vjp(f: &MultiaryFn, primals: &[UniquePtr<array>], cotangents: &[UniquePtr<array>]) -> [UniquePtr<CxxVector<array>>; 2];
    }

    // TODO: this needs to be placed in a separate file
    extern "Rust" {
        #[namespace = "mlx_cxx"]
        type DynFn;

        #[namespace = "mlx_cxx"]
        fn execute_dyn_fn(f: &DynFn, args: i32) -> i32;
    }
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn execute_fun_in_rust() {
//         use crate::transforms::Function;

//         let f = |x: i32| x + 1;
//         let y = f.execute(1);
//         assert_eq!(y, 2);
//     }

//     #[test]
//     fn execute_dyn_fun_callback() {
//         use super::DynFn;

//         // Define a type that is not `Copy` or `Clone`.
//         struct Bar {
//             s: String,
//         }

//         let bar = Bar { s: String::from("0123456") };
//         let f = move |x: i32| {
//             if x > 0 {
//                 x + 1
//             } else {
//                 bar.s.len() as i32
//             }
//         };
//         let f = Box::new(f) as DynFn;
//         let y = super::ffi::execute_callback(&f, 1);
//         assert_eq!(y, 2);

//         let y2 = super::ffi::execute_callback(&f, -1);
//         assert_eq!(y2, 7);
//     }
// }