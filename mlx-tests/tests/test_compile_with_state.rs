//! Tests for compilation of modules and optimizers.

mod common;

use common::LinearFunctionModel;
use mlx_rs::{
    assert_array_eq,
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::ones,
    optimizers::{Optimizer, Sgd},
    transforms::{compile::compile_with_state, eval_params},
    Array,
};

#[test]
fn test_compile_module() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Array {
        let y = model.forward(x).unwrap();
        y.square().unwrap().sum(None).unwrap()
    };
    let mut model = LinearFunctionModel::new(None).unwrap();

    let x = ones::<f32>(&[10, 1]).unwrap();
    let x = vec![x];

    let step = move |model: &mut LinearFunctionModel, x: &[Array]| -> Vec<Array> {
        let mut lg = nn::value_and_grad(loss);
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

fn compile_module_and_optimizer<O: Optimizer>(optimizer: O) -> (Array, Array) {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Array {
        let y = model.forward(x).unwrap();
        y.square().unwrap().sum(None).unwrap()
    };
    let model = LinearFunctionModel::new(None).unwrap();

    let x = ones::<f32>(&[10, 1]).unwrap();

    let step = move |(model, optimizer): &mut (LinearFunctionModel, O), x: &Array| -> Array {
        let mut lg = nn::value_and_grad(loss);
        let (loss, grad) = lg(model, x).unwrap();
        optimizer.update(model, grad).unwrap();
        loss
    };

    let mut state = (model, optimizer);
    let mut compiled = compile_with_state(step, None);

    let original = step(&mut state, &x);
    let result = compiled(&mut state, &x).unwrap();

    (original, result)
}

/// A simple sanity check for adafactor optimizer
#[test]
fn test_compile_module_and_adafactor_works() {
    let optimizer = mlx_rs::optimizers::Adafactor::new().unwrap();
    let (original, result) = compile_module_and_optimizer(optimizer);

    assert_eq!(original.shape(), result.shape());
    assert_eq!(original.dtype(), result.dtype());
}

#[test]
fn test_compile_module_and_sgd_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = Sgd::new(0.0);
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_and_adam_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = mlx_rs::optimizers::Adam::new(0.0);
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_and_rmsprop_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = mlx_rs::optimizers::RmsProp::new(0.0).unwrap();
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_and_adagrad_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = mlx_rs::optimizers::AdaGrad::new(0.0);
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_and_adadelta_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = mlx_rs::optimizers::AdaDelta::new(0.0).unwrap();
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_and_adamw_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = mlx_rs::optimizers::AdamW::new(0.0);
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_and_adamax_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = mlx_rs::optimizers::Adamax::new(0.0);
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_and_lion_consistency() {
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = mlx_rs::optimizers::Lion::new(0.0);
    let (original, result) = compile_module_and_optimizer(optimizer);
    assert_array_eq!(&original, &result);
}

#[test]
fn test_compile_module_with_error() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Result<Array, Exception> {
        let y = model.forward(x)?;
        y.square()?.sum(None)
    };
    let mut model = LinearFunctionModel::new(&[10]).unwrap();

    let step =
        move |model: &mut LinearFunctionModel, x: &[Array]| -> Result<Vec<Array>, Exception> {
            let mut lg = nn::value_and_grad(loss);
            let x = &x[0];
            let (loss, _grad) = lg(model, x)?;
            Ok(vec![loss])
        };

    // Make sure the compiled function produces the same result
    let mut compiled = compile_with_state(step, None);

    // input with correct shape
    let x_ok = ones::<f32>(&[10, 1]).unwrap();
    let x_ok = vec![x_ok];
    // input with wrong shape
    let x_err = ones::<f32>(&[1, 2, 3]).unwrap();
    let x_err = vec![x_err];

    // Success case
    // Check that the original function works
    let original = step(&mut model, x_ok.as_slice()).unwrap();

    let result = compiled(&mut model, x_ok.as_slice()).unwrap();
    assert_eq!(&original, &result);
    let result = compiled(&mut model, x_ok.as_slice()).unwrap();
    assert_eq!(&original, &result);

    // Error case

    // Check that the original function returns an error
    let original = step(&mut model, x_err.as_slice());
    assert!(original.is_err());
    // Make sure the compiled function also returns an error
    let result = compiled(&mut model, x_err.as_slice());
    assert!(result.is_err());
}

#[test]
fn test_compile_module_and_optimizer_with_error() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Result<Array, Exception> {
        let y = model.forward(x)?;
        y.square()?.sum(None)
    };
    let model = LinearFunctionModel::new(&[10]).unwrap();
    // Use a learning rate of 0.0 so that the parameters don't change
    // and we can check that the compiled function produces the same result
    let optimizer = Sgd::new(0.0);

    let step = move |(model, optimizer): &mut (LinearFunctionModel, Sgd),
                     x: &[Array]|
          -> Result<Vec<Array>, Exception> {
        let mut lg = nn::value_and_grad(loss);
        let x = &x[0];
        let (loss, grad) = lg(model, x)?;
        optimizer.update(model, grad)?;
        Ok(vec![loss])
    };

    let mut state = (model, optimizer);
    let mut compiled = compile_with_state(step, None);

    // input with correct shape
    let x_ok = ones::<f32>(&[10, 1]).unwrap();
    let x_ok = vec![x_ok];
    // input with wrong shape
    let x_err = ones::<f32>(&[1, 2, 3]).unwrap();
    let x_err = vec![x_err];

    // Success case
    // Check that the original function works
    let original = step(&mut state, x_ok.as_slice()).unwrap();

    let result = compiled(&mut state, x_ok.as_slice()).unwrap();
    assert_array_eq!(&original[0], &result[0]);
    eval_params(state.0.parameters()).unwrap();
    let result = compiled(&mut state, x_ok.as_slice()).unwrap();
    assert_array_eq!(&original[0], &result[0]);
    eval_params(state.0.parameters()).unwrap();

    // Error case

    // Check that the original function returns an error
    let original = step(&mut state, x_err.as_slice());
    assert!(original.is_err());
    // Make sure the compiled function also returns an error
    let result = compiled(&mut state, x_err.as_slice());
    assert!(result.is_err());
}
