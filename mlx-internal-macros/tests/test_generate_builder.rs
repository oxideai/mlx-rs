use mlx_internal_macros::generate_builder;

generate_builder! {
    /// Test struct for the builder generation.
    #[derive(Debug)]
    #[generate_builder(generate_build_fn = true)]
    struct TestStruct {
        #[optional(default_value = TestStruct::DEFAULT_OPT_FIELD_1)]
        opt_field_1: i32,
        #[optional(default_value = TestStruct::DEFAULT_OPT_FIELD_2)]
        opt_field_2: i32,
        mandatory_field_1: i32,
        mandatory_field_2: i32,
    }
}

impl TestStruct {
    pub const DEFAULT_OPT_FIELD_1: i32 = 1;
    pub const DEFAULT_OPT_FIELD_2: i32 = 2;
}

#[test]
fn build_test_struct() {
    let test_struct = TestStruct::builder()
        .opt_field_1(2)
        .opt_field_2(3)
        .build(4, 5);

    assert_eq!(test_struct.opt_field_1, 2);
    assert_eq!(test_struct.opt_field_2, 3);
    assert_eq!(test_struct.mandatory_field_1, 4);
    assert_eq!(test_struct.mandatory_field_2, 5);
}
