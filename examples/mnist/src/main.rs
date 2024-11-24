use mlx_nn::{
    losses::{CrossEntropyBuilder, LossReduction},
    module_value_and_grad,
};
use mlx_rs::{
    array,
    error::Exception,
    module::{Module, ModuleParameters},
    optimizers::{Optimizer, Sgd},
    transforms::eval_params,
    Array,
    builder::Builder
};

/// MLP model
mod mlp;

/// Retrieves MNIST dataset
mod mnist;

#[derive(Clone)]
struct Loader {}

impl Iterator for Loader {
    type Item = (Array, Array);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn load_training_data() -> Result<Loader, Box<dyn std::error::Error>> {
    todo!()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_layers = 3;
    let input_dim = 784;
    let hidden_dim = 256;
    let output_dim = 10;
    let lr = 1e-2;
    let num_epochs = 10;

    let loader = load_training_data()?;
    let mut model = mlp::Mlp::new(num_layers, input_dim, hidden_dim, output_dim)?;

    let cross_entropy = CrossEntropyBuilder::new()
        .reduction(LossReduction::Mean)
        .build()?;
    let loss_fn = |model: &mut mlp::Mlp, (x, y): (&Array, &Array)| -> Result<Array, Exception> {
        let y_pred = model.forward(x)?;
        cross_entropy.apply(y_pred, y)
    };
    let mut loss_and_grad_fn = module_value_and_grad(loss_fn);

    let mut optimizer = Sgd::new(lr);

    for _ in 0..num_epochs {
        let mut loss_sum = array!(0.0);
        for (x, y) in loader.clone() {
            let (loss, grad) = loss_and_grad_fn(&mut model, (&x, &y))?;
            optimizer.apply(&mut model, grad).unwrap();
            eval_params(model.parameters())?;

            loss_sum += loss;
        }

        println!("Epoch: {}, Loss sum: {}", num_epochs, loss_sum);
    }

    Ok(())
}
