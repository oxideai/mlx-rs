use mlx_nn::{
    losses::{CrossEntropyOptions, LossReduction},
    module_value_and_grad,
    optimizers::Optimizer,
};
use mlx_rs::{
    array,
    error::Exception,
    module::{Module, ModuleParameters},
    transforms::eval_params,
    Array,
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

    let loss_fn = |model: &mlp::Mlp, (x, y): (&Array, &Array)| -> Result<Array, Exception> {
        let y_pred = model.forward(x)?;

        // Config optional parameters for cross entropy loss
        let options = CrossEntropyOptions::builder()
            .reduction(LossReduction::Mean)
            .build();
        mlx_nn::losses::cross_entropy(y_pred, y, options)
    };
    let mut loss_and_grad_fn = module_value_and_grad(loss_fn);

    let mut optimizer = mlx_nn::optimizers::Sgd::new(lr);

    for _ in 0..num_epochs {
        let mut loss_sum = array!(0.0);
        for (x, y) in loader.clone() {
            let (loss, grad) = loss_and_grad_fn(&mut model, (&x, &y))?;
            optimizer.update(&mut model, grad);
            eval_params(model.parameters())?;

            loss_sum += loss;
        }

        println!("Epoch: {}, Loss sum: {}", num_epochs, loss_sum);
    }

    Ok(())
}
