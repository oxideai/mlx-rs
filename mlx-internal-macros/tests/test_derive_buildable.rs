use mlx_internal_macros::Buildable;

const DEFAULT_A: i32 = 10;

#[derive(Buildable)]
struct TestExample {
    #[buildable(optional, default = DEFAULT_A)]
    a: i32,
    b: i32,
    c: i32,
}
