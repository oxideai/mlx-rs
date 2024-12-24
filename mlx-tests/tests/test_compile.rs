//! Tests for compilation of modules and optimizers.

mod common;

use common::LinearFunctionModel;
use mlx_nn::module_value_and_grad;
use mlx_rs::{array, module::Module, random::uniform, transforms::compile::compile, Array};



#[test]
fn test_compile_module() {
    let loss = |model: &mut LinearFunctionModel, x: &Array| -> Array {
        let y = model.forward(x).unwrap();
        y.square().unwrap().sum(None, None).unwrap()
    };
    let mut model = LinearFunctionModel::new().unwrap();

    let m = array!(0.25);
    let b = array!(0.75);

    
    let step = move |(model, x): (&mut LinearFunctionModel, &[Array])| -> Vec<Array> {
        let mut lg = module_value_and_grad(loss);
        let x = &x[0];
        let (loss, grad) = lg(model, x).unwrap();
        println!("loss: {:?}", loss);
        vec![loss]
    };

    let mut compiled_step = compile(step, None);

    let x = uniform::<_, f32>(-5.0, 5.0, &[10, 1], None).unwrap();
    let x = vec![x];
    compiled_step((&mut model, &x));
}