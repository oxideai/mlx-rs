# Mistral

An example of generating text with Mistral-7B-v0.1 model using mlx-rs.

This is the rust version of the [mlx-examples/llms/mistral](https://github.com/ml-explore/mlx-examples/tree/main/llms/mistral) example.

## Usage

This example loads the safetensors version of the model from a huggingface repo and thus requires internet connection the first time it is run.

To run the example in release mode, execute the following command:

```bash
cargo run --release
```

### Arguments

The example accepts the following optional arguments:

- `--prompt: str` - The message to be processed by the model. Default: "In the beginning the Universe was created."
- `--max-tokens: int` - The maximum number of tokens to generate. Default: 100
- `--temp: float` - The sampling temperature. Default: 0.0
- `--tokens-per-eval: int` - The batch size of tokens to generate. Default: 10
- `--seed: int` - The PRNG seed. Default: 0

For example, to generate text with a prompt "Hello, world!" and a seed of 1 (in release mode), run the following command:

```bash
cargo run --release -- --prompt "Hello, world!" --seed 1
```
