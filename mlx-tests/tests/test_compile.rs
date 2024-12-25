//! Tests for compilation of modules and optimizers.

mod common;

use common::LinearFunctionModel;
use mlx_nn::module_value_and_grad;
use mlx_rs::{array, assert_array_eq, module::{Module, ModuleParameters}, ops::ones, optimizers::{Optimizer, Sgd}, random::uniform, transforms::{compile::{compile, compile_with_state}, eval_params}, Array};



#[test]
fn test_compile_with_module() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Array {
        let y = model.forward(x).unwrap();
        y.square().unwrap().sum(None, None).unwrap()
    };
    let mut model = LinearFunctionModel::new().unwrap();

    let x = ones::<f32>(&[10, 1]).unwrap();
    let x = vec![x];

    let step = move |model: &mut LinearFunctionModel, x: &[Array]| -> Vec<Array> {
        let mut lg = module_value_and_grad(loss);
        let x = &x[0];
        let (loss, _grad) = lg(model, x).unwrap();
        vec![loss]
    };

    // Check that the original function works
    let original = step(&mut model, x.as_slice());

    // Make sure the compiled function produces the same result
    let mut compiled = compile_with_state(step, None);
    let result = compiled(&mut model, x.as_slice()).unwrap();
    assert_eq!(&original, &result);
    let result = compiled(&mut model, x.as_slice()).unwrap();
    assert_eq!(&original, &result);
}

#[test]
fn test_compile_with_module_and_optimizer() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Array {
        let y = model.forward(x).unwrap();
        y.square().unwrap().sum(None, None).unwrap()
    };
    let model = LinearFunctionModel::new().unwrap();
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = Sgd::new(0.0);

    let x = ones::<f32>(&[10, 1]).unwrap();
    let x = vec![x];

    let step = move |(model, optimizer): &mut (LinearFunctionModel, Sgd), x: &[Array]| -> Vec<Array> {
        let mut lg = module_value_and_grad(loss);
        let x = &x[0];
        let (loss, grad) = lg(model, x).unwrap();
        optimizer.update(model, grad).unwrap();
        vec![loss]
    };

    let mut state = (model, optimizer);
    let mut compiled = compile_with_state(step, None);
    
    // Check that the original function works
    let original = step(&mut state, x.as_slice());

    // Make sure the compiled function produces the same result
    let result = compiled(&mut state, x.as_slice()).unwrap();
    assert_array_eq!(&original[0], &result[0]);
    eval_params(state.0.parameters()).unwrap();
    let result = compiled(&mut state, x.as_slice()).unwrap();
    assert_array_eq!(&original[0], &result[0]);
    eval_params(state.0.parameters()).unwrap();
}