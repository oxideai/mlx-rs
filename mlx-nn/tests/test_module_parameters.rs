use mlx_macros::ModuleParameters;
use mlx_rs::module::{ModuleParameters, Param, Parameter};
use mlx_rs::{array, Array};

#[derive(ModuleParameters)]
pub struct StructModule {
    #[param]
    a: Param<Array>,

    #[param]
    b: Param<Array>,

    #[param]
    c: Param<Option<Array>>,
}

#[derive(ModuleParameters)]
pub struct UnitStructModule;

#[test]
fn test_module_parameters() {
    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(Some(array!(3.0))),
    };

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));
    assert_eq!(flattened["c"], &array!(3.0));
}

#[test]
fn test_module_parameters_mut() {
    let mut m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &mut array!(1.0));
    assert_eq!(flattened["b"], &mut array!(2.0));

    let mut m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(Some(array!(3.0))),
    };

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened["a"], &mut array!(1.0));
    assert_eq!(flattened["b"], &mut array!(2.0));
    assert_eq!(flattened["c"], &mut array!(3.0));
}

#[test]
fn test_module_trainable_parameters_all_trainable() {
    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    let m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(Some(array!(3.0))),
    };

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));
    assert_eq!(flattened["c"], &array!(3.0));
}

#[test]
fn test_module_trainable_parameters_partial_freeze() {
    let mut m = StructModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    // Freeze one parameter that is not optional
    m.a.freeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 1);
    assert_eq!(flattened["b"], &array!(2.0));

    // Now freeze the optional parameter
    m.c.freeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 1);
    assert_eq!(flattened["b"], &array!(2.0));

    // Unfreeze the non-optional parameter
    m.a.unfreeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    // Set the optional parameter to Some but still frozen
    m.c.value = Some(array!(3.0));

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    // Unfreeze the optional parameter
    m.c.unfreeze(true);

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));
    assert_eq!(flattened["c"], &array!(3.0));
}

#[test]
fn test_unit_struct_module_parameters() {
    let m = UnitStructModule;

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 0);
}

#[test]
fn test_unit_struct_module_parameters_mut() {
    let mut m = UnitStructModule;

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 0);
}

#[test]
fn test_unit_struct_module_trainable_parameters() {
    let m = UnitStructModule;

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 0);
}

#[derive(ModuleParameters)]
struct StructModuleWithNested {
    #[param]
    a: Param<Array>,

    #[param]
    nested: StructModule,
}

#[test]
fn test_nested_module_parameters() {
    let m = StructModuleWithNested {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
    };

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["nested.a"], &array!(2.0));
    assert_eq!(flattened["nested.b"], &array!(3.0));
}

#[test]
fn test_nested_module_parameters_mut() {
    let mut m = StructModuleWithNested {
        a: Param::new(array!(1.0)),
        nested: StructModule {
            a: Param::new(array!(2.0)),
            b: Param::new(array!(3.0)),
            c: Param::new(None),
        },
    };

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened["a"], &mut array!(1.0));
    assert_eq!(flattened["nested.a"], &mut array!(2.0));
    assert_eq!(flattened["nested.b"], &mut array!(3.0));
}
