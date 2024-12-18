use mlx_rs::{nn::{
    losses::{CrossEntropyBuilder, LossReduction},
    module_value_and_grad,
}, transforms::async_eval_params};
use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::{Module, ModuleParameters},
    ops::{eq, indexing::argmax, mean},
    optimizers::{Optimizer, Sgd},
    transforms::eval_params,
    Array,
};

/// MLP model
mod mlp;

/// Retrieves MNIST dataset
mod data;

fn eval_fn(model: &mut mlp::Mlp, (x, y): (&Array, &Array)) -> Result<Array, Exception> {
    let y_pred = model.forward(x)?;
    let accuracy = mean(&eq(&argmax(&y_pred, 1, None)?, y)?, None, None)?;
    Ok(accuracy)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_layers = 2;
    let hidden_dim = 32;
    let num_classes = 10;
    let batch_size = 256;
    let num_epochs = 10;
    let learning_rate = 1e-1;

    let (train_images, train_labels, test_images, test_labels) = data::read_data();
    let loader = data::iterate_data(&train_images, &train_labels, batch_size)?;

    let input_dim = train_images[0].shape()[0];
    let mut model = mlp::Mlp::new(num_layers, input_dim, hidden_dim, num_classes)?;

    let cross_entropy = CrossEntropyBuilder::new()
        .reduction(LossReduction::Mean)
        .build()?;
    let loss_fn = |model: &mlp::Mlp, (x, y): (&Array, &Array)| -> Result<Array, Exception> {
        let y_pred = model.forward(x)?;
        cross_entropy.apply(y_pred, y)
    };
    let mut loss_and_grad_fn = module_value_and_grad(loss_fn);

    let mut optimizer = Sgd::new(learning_rate);

    println!("{:?}", model.parameters().flatten()["layers.layers.0.bias"]);

    let accuracy = eval_fn(&mut model, (&test_images, &test_labels))?;
    println!("Initial accuracy: {:?}", accuracy.item::<f32>());

    for e in 0..num_epochs {
        let now = std::time::Instant::now();
        let mut lg_time = std::time::Duration::from_secs(0);
        let mut apply_time = std::time::Duration::from_secs(0);
        let mut eval_time = std::time::Duration::from_secs(0);
        let mut train_loss = mlx_rs::array!(0.0);
        for (x, y) in &loader {
            let lg_start = std::time::Instant::now();
            let (loss, grad) = loss_and_grad_fn(&model, (x, y))?;
            lg_time += lg_start.elapsed();

            let apply_start = std::time::Instant::now();
            optimizer.apply(&model, grad).unwrap();
            apply_time += apply_start.elapsed();

            let eval_start = std::time::Instant::now();
            async_eval_params(model.parameters())?;
            eval_time += eval_start.elapsed();

            train_loss += loss;
        }

        // Evaluate on test set
        let accuracy = eval_fn(&mut model, (&test_images, &test_labels))?;
        let elapsed = now.elapsed();

        println!(
            "Epoch: {}, Test accuracy: {:?}, Time: {:?}",
            e,
            accuracy.item::<f32>(),
            elapsed
        );

        println!(
            "Loss and grad time: {:?} s, Apply time: {:?}, Eval time: {:?}, train loss: {:?}",
            lg_time,
            apply_time,
            eval_time,
            train_loss
        );
    }

    println!("{:?}", model.parameters().flatten()["layers.layers.0.bias"]);

    Ok(())
}
