use mlx_internal_macros::*;
use mlx_rs::builder::{Buildable, Builder};

generate_builder! {
    /// Test struct for the builder generation.
    #[derive(Debug, Buildable)]
    #[builder(manual_impl)]
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

impl Builder<TestStruct> for TestStructBuilder {
    type Error = std::convert::Infallible;

    fn build(self) -> std::result::Result<TestStruct, Self::Error> {
        Ok(TestStruct {
            opt_field_1: self.opt_field_1,
            opt_field_2: self.opt_field_2,
            mandatory_field_1: self.mandatory_field_1,
            ignored_field: String::from("ignored"),
        })
    }
}

impl TestStruct {
    pub const DEFAULT_OPT_FIELD_1: i32 = 1;
    pub const DEFAULT_OPT_FIELD_2: i32 = 2;
}

#[test]
fn build_test_struct() {
    let test_struct = <TestStruct as Buildable>::Builder::new(4)
        .opt_field_1(2)
        .opt_field_2(3)
        .build().unwrap();

    assert_eq!(test_struct.opt_field_1, 2);
    assert_eq!(test_struct.opt_field_2, 3);
    assert_eq!(test_struct.mandatory_field_1, 4);
    assert_eq!(test_struct.ignored_field, String::from("ignored"));
}
