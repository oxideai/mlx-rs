//! Tests for compilation of modules and optimizers.

mod common;

use common::LinearFunctionModel;
use mlx_nn::module_value_and_grad;
use mlx_rs::{array, module::{Module, ModuleParameters}, ops::ones, optimizers::{Optimizer, Sgd}, random::uniform, transforms::{compile::{compile, compile_with_state}, eval_params}, Array};



#[test]
fn test_compile_module() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Array {
        let y = model.forward(x).unwrap();
        y.square().unwrap().sum(None, None).unwrap()
    };
    let mut model = LinearFunctionModel::new().unwrap();
    let mut optimizer = Sgd::new(1e-2);

    let m = array!(0.25);
    let b = array!(0.75);
    // let x = uniform::<_, f32>(-5.0, 5.0, &[10, 1], None).unwrap();
    let x = ones::<f32>(&[10, 1]).unwrap();
    let y = m * &x + b;
    let x = vec![x];

    let step = move |model: &mut LinearFunctionModel, x: &[Array]| -> Vec<Array> {
        let mut lg = module_value_and_grad(loss);
        let x = &x[0];
        let (loss, grad) = lg(model, x).unwrap();
        // optimizer.update(model, grad).unwrap();
        // println!("loss: {:?}", loss);
        vec![loss]
    };

    let original = step(&mut model, x.as_slice());
    println!("original: {:?}", original);

    let mut compiled = compile_with_state(step, None);
    let result = compiled(&mut model, x.as_slice());
    println!("result: {:?}", result);
    let result = compiled(&mut model, x.as_slice());
    println!("result: {:?}", result);

    // let step = move |(model, optimizer): &mut (&mut LinearFunctionModel, &mut Sgd), x: &[Array]| -> Vec<Array> {
    //     let mut lg = module_value_and_grad(loss);
    //     let x = &x[0];
    //     let (loss, grad) = lg(model, x).unwrap();
    //     optimizer.update(model, grad).unwrap();
    //     // println!("loss: {:?}", loss);
    //     vec![loss]
    // };

    // let mut compiled = compile_with_state(step, None);
    // let mut state = (&mut model, &mut optimizer);
    // let result = compiled(&mut state, x.as_slice());
    // eval_params(state.0.parameters()).unwrap();
    // let result = compiled(&mut state, x.as_slice());
    // eval_params(state.0.parameters()).unwrap();
}