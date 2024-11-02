//! Tests for the optimizers. These tests are placed here because the models
//! used for testing make use of `ModuleParameter` macro.

use mlx_rs::{
    array, assert_array_eq,
    error::Exception,
    losses::{LossReduction, MseLoss},
    module::{FlattenedModuleParam, Module, ModuleParameters, Param},
    ops::{ones, zeros},
    optimizers::{AdaDelta, AdaGrad, Adam, AdamW, Adamax, Optimizer, RmsProp, Sgd},
    random::uniform,
    transforms::{eval, eval_params},
    Array, Dtype,
};

use mlx_nn::{macros::ModuleParameters, module_value_and_grad};

/* -------------------------------------------------------------------------- */
/*                              Convergence tests                             */
/* -------------------------------------------------------------------------- */

/// A helper model for testing optimizers.
///
/// This is adapted from the swift binding tests in `mlx-swift/Tests/MLXTests/OptimizerTests.swift`.
#[derive(Debug, ModuleParameters)]
struct LinearFunctionModel {
    #[param]
    pub m: Param<Array>,

    #[param]
    pub b: Param<Array>,
}

impl Module for LinearFunctionModel {
    type Error = Exception;

    fn forward(&self, x: &Array) -> Result<Array, Self::Error> {
        self.m.multiply(x)?.add(&self.b)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

impl LinearFunctionModel {
    pub fn new() -> Result<Self, Exception> {
        let m = uniform::<_, f32>(-5.0, 5.0, None, None)?;
        let b = uniform::<_, f32>(-5.0, 5.0, None, None)?;
        Ok(Self {
            m: Param::new(m),
            b: Param::new(b),
        })
    }
}

pub fn train<F, O>(f: F, steps: usize) -> Result<Array, Box<dyn std::error::Error>>
where
    F: FnOnce() -> O,
    O: Optimizer,
{
    let mut optimizer = f();

    let mse_loss = MseLoss::builder().reduction(LossReduction::Mean).build();
    let loss = |model: &LinearFunctionModel, (x, y): (&Array, &Array)| {
        mse_loss.apply(model.forward(x)?, y)
    };

    // TODO: check compiled model once we have it
    let mut model = LinearFunctionModel::new()?;
    eval_params(model.parameters())?;

    let m = array!(0.25);
    let b = array!(7.0);

    let mut lg = module_value_and_grad(loss);

    let mut last_loss = None;
    for _ in 0..steps {
        // println!("target: b = {}, m = {}", b, m);
        // println!("parameters: {:?}", model.parameters());

        // generate random training data along with the ground truth.
        // notice that the shape is [B, 1] where B is the batch
        // dimension -- this allows us to train on 10 samples simultaneously
        let x = uniform::<_, f32>(-5.0, 5.0, &[10, 1], None)?;
        let y = &m * &x + &b;
        eval([&x, &y])?;

        // compute the loss and gradients.  use the optimizer
        // to adjust the parameters closer to the target
        let (loss, g) = lg(&mut model, (&x, &y))?;
        optimizer.apply(&mut model, g)?;

        eval_params(model.parameters())?;

        last_loss = Some(loss);
    }

    Ok(last_loss.unwrap())
}

const NUM_TRIALS: usize = 3;

#[test]
fn test_sgd_converges() {
    let mut total_loss = 0.0;
    for _ in 0..NUM_TRIALS {
        let loss = train(|| Sgd::new(0.1), 30).unwrap();
        total_loss += loss.item::<f32>();
    }
    // It sometimes doesn't converge that fast, so we take the average loss
    // across multiple trials
    let avg_loss = total_loss / NUM_TRIALS as f32;
    assert!(avg_loss < 0.1, "avg loss: {}", avg_loss);
}

#[test]
fn test_rmsprop_converges() {
    let mut total_loss = 0.0;
    for _ in 0..NUM_TRIALS {
        // RMSProp doesn't seem to converge as fast as SGD
        let loss = train(|| RmsProp::new(0.1), 100).unwrap();
        total_loss += loss.item::<f32>();
    }
    // It sometimes doesn't converge that fast, so we take the average loss
    // across multiple trials
    let avg_loss = total_loss / NUM_TRIALS as f32;
    assert!(avg_loss < 0.1, "avg loss: {}", avg_loss);
}

/* -------------------------------------------------------------------------- */
/*                            Optimizer unit tests                            */
/* -------------------------------------------------------------------------- */

#[derive(Debug, ModuleParameters)]
struct SimpleModel {
    #[param]
    a: Param<Array>,
}

#[derive(Debug, ModuleParameters)]
struct First {
    #[param]
    pub a: Param<Array>,

    #[param]
    pub b: Param<Array>,
}

#[derive(Debug, ModuleParameters)]
struct NestedModel {
    #[param]
    pub first: Param<First>,

    #[param]
    pub second: Param<Array>,
}

type GradsMap = FlattenedModuleParam;

fn create_default_test_model_and_grads() -> (NestedModel, GradsMap) {
    let first = First {
        a: Param::new(zeros::<f32>(&[10]).unwrap()),
        b: Param::new(zeros::<f32>(&[1]).unwrap()),
    };
    let model = NestedModel {
        first: Param::new(first),
        second: Param::new(zeros::<f32>(&[1]).unwrap()),
    };

    let grads_map: GradsMap = model
        .parameters()
        .flatten()
        .iter()
        .map(|(k, v)| {
            let g = ones::<f32>(v.shape()).unwrap();
            (k.clone(), g)
        })
        .collect();

    (model, grads_map)
}

const ATOL: f64 = 1e-5;

// This unit test is adapted from the swift binding unit test `testAdaDelta` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_ada_delta() {
    mlx_rs::random::seed(547);
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), mlx_rs::Dtype::Float32);
    assert_array_eq!(
        a.mean(None, None).unwrap(),
        array!(-0.348_337_02),
        0.006966740489006043
    );
    assert_array_eq!(
        a.sum(None, None).unwrap(),
        array!(-4.180_044),
        0.08360088348388672
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), mlx_rs::Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, None).unwrap(),
        array!(0.522_678_4),
        0.010453567504882813
    );
    assert_array_eq!(
        a_grad.sum(None, None).unwrap(),
        array!(6.272_14),
        0.12544280052185058
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdaDelta::new(0.1);

    optimizer.apply(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), mlx_rs::Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, None).unwrap(),
        array!(-0.348_442_4),
        0.348442405462265
    );
    assert_array_eq!(
        a_model.a.sum(None, None).unwrap(),
        array!(-4.181_308_7),
        0.08362617492675782
    );
}

// This unit test is adapted from the swift binding unit test `testAdaGrad` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adagrad() {
    mlx_rs::random::seed(958);
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(a.mean(None, None).unwrap(), array!(-0.045_843_333), ATOL);
    assert_array_eq!(a.sum(None, None).unwrap(), array!(-0.550_12), ATOL);

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(a_grad.mean(None, None).unwrap(), array!(0.232_503_94), ATOL);
    assert_array_eq!(a_grad.sum(None, None).unwrap(), array!(2.790_047_2), ATOL);

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdaGrad::new(0.1);

    optimizer.apply(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, None).unwrap(),
        array!(-0.062_509_984),
        ATOL
    );
    assert_array_eq!(
        a_model.a.sum(None, None).unwrap(),
        array!(-0.750_119_8),
        ATOL
    );
}

// This unit test is adapted from the swift binding unit test `testAdam` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adam() {
    mlx_rs::random::seed(616);
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, None).unwrap(),
        array!(0.112_293_06),
        0.002245861142873764
    );
    assert_array_eq!(
        a.sum(None, None).unwrap(),
        array!(1.347_516_7),
        0.02695033311843872
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, None).unwrap(),
        array!(0.305_597_72),
        0.0061119544506073
    );
    assert_array_eq!(
        a_grad.sum(None, None).unwrap(),
        array!(3.667_172_7),
        0.0733434534072876
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Adam::new(0.1);

    optimizer.apply(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, None).unwrap(),
        array!(0.112_292_78),
        0.0022458556294441224
    );
    assert_array_eq!(
        a_model.a.sum(None, None).unwrap(),
        array!(1.347_513_3),
        0.026950266361236572
    );
}

// This unit test is adapted from the swift binding unit test `testAdamW` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adamw() {
    mlx_rs::random::seed(696);
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, None).unwrap(),
        array!(-0.363_391_88),
        0.007267837524414063
    );
    assert_array_eq!(
        a.sum(None, None).unwrap(),
        array!(-4.360_702_5),
        0.08721405029296875
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, None).unwrap(),
        array!(0.221_754_48),
        0.0044350895285606385
    );
    assert_array_eq!(
        a_grad.sum(None, None).unwrap(),
        array!(2.661_053_7),
        0.05322107315063477
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdamW::new(0.1);

    optimizer.apply(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, None).unwrap(),
        array!(-0.468_437_6),
        0.009368752241134645
    );
    assert_array_eq!(
        a_model.a.sum(None, None).unwrap(),
        array!(-5.621_251),
        0.11242502212524415
    );
}

// This unit test is adapted from the python unit test `test_adamax` in
// `mlx/python/tests/test_optimizers.py`.
#[test]
fn test_adamax() {
    mlx_rs::random::seed(75);
    let a = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, None).unwrap(),
        array!(-0.303_923_6),
        0.006078472137451172
    );
    assert_array_eq!(
        a.sum(None, None).unwrap(),
        array!(-3.647_083_3),
        0.07294166564941407
    );

    let a_grad = mlx_rs::random::normal::<f32>(&[4, 3], None, None, None).unwrap();
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, None).unwrap(),
        array!(-0.242_717_24),
        0.004854344725608826
    );
    assert_array_eq!(
        a_grad.sum(None, None).unwrap(),
        array!(-2.912_606_7),
        0.05825213432312012
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Adamax::new(0.1);

    optimizer.apply(&mut a_model, a_grad_params).unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, None).unwrap(),
        array!(-0.303_923_6),
        0.006078472137451172
    );
    assert_array_eq!(
        a_model.a.sum(None, None).unwrap(),
        array!(-3.647_083_3),
        0.07294166564941407
    );
}

// This unit test is adapted from the python unit test `test_rmsprop` in
// `tests/test_optimizer.py`.
#[test]
fn test_rmsprop() {
    const LR: f32 = 1e-2;
    const ALPHA: f32 = 0.99;

    let (mut model, gradients) = create_default_test_model_and_grads();

    let mut optim = RmsProp::builder().alpha(ALPHA).build(LR).unwrap();
    optim.apply(&mut model, gradients).unwrap();

    let expected_first_a = ones::<f32>(&[10]).unwrap() * -0.1;
    let expected_first_b = ones::<f32>(&[1]).unwrap() * -0.1;
    let expected_second = ones::<f32>(&[1]).unwrap() * -0.1;

    assert_array_eq!(model.first.a.as_ref(), expected_first_a, ATOL);
    assert_array_eq!(model.first.b.as_ref(), expected_first_b, ATOL);
    assert_array_eq!(model.second.as_ref(), expected_second, ATOL);

    let expected_state_first_a = ones::<f32>(&[10]).unwrap() * 0.01;
    let expected_state_first_b = ones::<f32>(&[1]).unwrap() * 0.01;
    let expected_state_second = ones::<f32>(&[1]).unwrap() * 0.01;

    assert_array_eq!(
        optim.state.get("first.a").unwrap(),
        expected_state_first_a,
        ATOL
    );
    assert_array_eq!(
        optim.state.get("first.b").unwrap(),
        expected_state_first_b,
        ATOL
    );
    assert_array_eq!(
        optim.state.get("second").unwrap(),
        expected_state_second,
        ATOL
    );
}

// This unit test is adapted from the python unit test `test_sgd` in
// `mlx/python/tests/test_optimizers.py`
#[test]
fn test_sgd() {
    let (mut model, gradients) = create_default_test_model_and_grads();

    let mut optim = Sgd::builder().momentum(0.9).build(1e-2);
    optim.apply(&mut model, gradients).unwrap();

    let expected_first_a = ones::<f32>(&[10]).unwrap() * -0.01;
    let expected_first_b = ones::<f32>(&[1]).unwrap() * -0.01;
    let expected_second = ones::<f32>(&[1]).unwrap() * -0.01;

    assert_array_eq!(model.first.a.as_ref(), expected_first_a, ATOL);
    assert_array_eq!(model.first.b.as_ref(), expected_first_b, ATOL);
    assert_array_eq!(model.second.as_ref(), expected_second, ATOL);

    let expected_state_first_a = ones::<f32>(&[10]).unwrap();
    let expected_state_first_b = ones::<f32>(&[1]).unwrap();
    let expected_state_second = ones::<f32>(&[1]).unwrap();

    assert_array_eq!(
        optim.state["first.a"].as_ref(),
        expected_state_first_a,
        ATOL
    );
    assert_array_eq!(
        optim.state["first.b"].as_ref(),
        expected_state_first_b,
        ATOL
    );
    assert_array_eq!(optim.state["second"].as_ref(), expected_state_second, ATOL);
}
