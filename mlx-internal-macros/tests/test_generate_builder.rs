use mlx_internal_macros::*;
use mlx_rs::builder::{Buildable, Builder};

generate_builder! {
    /// Test struct for the builder generation.
    #[derive(Debug, Buildable)]
    #[builder(build_with = build_test_struct)]
    struct TestStruct {
        #[builder(optional, default = TestStruct::DEFAULT_OPT_FIELD_1)]
        opt_field_1: i32,
        #[builder(optional, default = TestStruct::DEFAULT_OPT_FIELD_2)]
        opt_field_2: i32,
        mandatory_field_1: i32,

        #[builder(ignore)]
        ignored_field: String,
    }
}

fn build_test_struct(
    builder: TestStructBuilder,
) -> std::result::Result<TestStruct, std::convert::Infallible> {
    Ok(TestStruct {
        opt_field_1: builder.opt_field_1,
        opt_field_2: builder.opt_field_2,
        mandatory_field_1: builder.mandatory_field_1,
        ignored_field: String::from("ignored"),
    })
}

impl TestStruct {
    pub const DEFAULT_OPT_FIELD_1: i32 = 1;
    pub const DEFAULT_OPT_FIELD_2: i32 = 2;
}

#[test]
fn test_generated_builder() {
    let test_struct = <TestStruct as Buildable>::Builder::new(4)
        .opt_field_1(2)
        .opt_field_2(3)
        .build()
        .unwrap();

    assert_eq!(test_struct.opt_field_1, 2);
    assert_eq!(test_struct.opt_field_2, 3);
    assert_eq!(test_struct.mandatory_field_1, 4);
    assert_eq!(test_struct.ignored_field, String::from("ignored"));
}
