<div align="center">
<h1><b>mlx-rs</b></h1>

Rust bindings for Apple's mlx machine learning library.

[![Discord](https://img.shields.io/discord/1176807732473495552.svg?color=7289da&&logo=discord)](https://discord.gg/jZvTsxDX49)
[![Current Crates.io Version](https://img.shields.io/crates/v/mlx-sys.svg)](https://crates.io/crates/mlx-rs)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://oxideai.github.io/mlx-rs/)
[![Test Status](https://github.com/oxideai/mlx-rs/actions/workflows/validate.yml/badge.svg)](https://github.com/oxideai/mlx-rs/actions/workflows/validate.yml)
[![Blaze](https://runblaze.dev/gh/307493885959233117281096297203102330146/badge.svg)](https://runblaze.dev)
[![Rust Version](https://img.shields.io/badge/Rust-1.75.0+-blue)](https://releases.rs/docs/1.75.0)
![license](https://shields.io/badge/license-MIT-blue)

> **⚠️ Project is still in development - contributors welcome!**

---

<div align="left" valign="middle">
<a href="https://runblaze.dev">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://www.runblaze.dev/logo_dark.png">
   <img align="right" src="https://www.runblaze.dev/logo_light.png" height="102px"/>
 </picture>
</a>

<br style="display: none;"/>

_[Blaze](https://runblaze.dev) supports this project by providing ultra-fast Apple Silicon macOS Github Action Runners. Apply the discount code `AI25` at checkout to enjoy 25% off your first year._

</div>

</div>

## Feature Flags

* `metal` - enables metal (GPU) usage in MLX
* `accelerate` - enables using the accelerate framework in MLX

## Versioning

For simplicity, the main crate `mls-rs` follows MLX’s versioning, allowing you to easily see which MLX version you’re using under the hood. The `mlx-sys` crate follows the versioning of `mlx-c`, as that is the version from which the API is generated. The `mlx-macros` crate uses its own versioning, as those macros are developed independently by us.

## Community

If you are excited about the project or want to contribute, don't hesitate to join our [Discord](https://discord.gg/jZvTsxDX49)!
We try to be as welcoming as possible to everybody from any background. We're still building this out, but you can ask your questions there!

## Status

mlx-rs is currently in active development, and is not yet complete.

## License

mlx-rs is distributed under the terms of the MIT license. See [LICENSE](./LICENSE) for details.
Opening a pull request is assumed to signal agreement with these licensing terms.
