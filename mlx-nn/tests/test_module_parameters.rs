use mlx_macros::ModuleParameters;
use mlx_rs::{array, Array};
use mlx_nn_module::{Param, ModuleParameters, Parameter};

#[derive(ModuleParameters)]
pub struct TestModule {
    #[param]
    a: Param<Array>,

    #[param]
    b: Param<Array>,

    #[param]
    c: Param<Option<Array>>,
}

#[test]
fn test_module_parameters() {
    let m = TestModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    let m = TestModule {
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
    let mut m = TestModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.parameters_mut().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &mut array!(1.0));
    assert_eq!(flattened["b"], &mut array!(2.0));

    let mut m = TestModule {
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
    let m = TestModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    let m = TestModule {
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
    let mut m = TestModule {
        a: Param::new(array!(1.0)),
        b: Param::new(array!(2.0)),
        c: Param::new(None),
    };

    // Freeze one parameter that is not optional
    m.a.freeze();

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 1);
    assert_eq!(flattened["b"], &array!(2.0));

    // Now freeze the optional parameter
    m.c.freeze();

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 1);
    assert_eq!(flattened["b"], &array!(2.0));

    // Unfreeze the non-optional parameter
    m.a.unfreeze();

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    // Set the optional parameter to Some but still frozen
    m.c.inner = Some(array!(3.0));

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 2);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));

    // Unfreeze the optional parameter
    m.c.unfreeze();

    let flattened = m.trainable_parameters().flatten();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened["a"], &array!(1.0));
    assert_eq!(flattened["b"], &array!(2.0));
    assert_eq!(flattened["c"], &array!(3.0));
}