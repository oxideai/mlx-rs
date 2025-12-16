# Changelog

## 0.25.3

- @dshan4585 Prevent premature destructuring of closures & Add atan2 (#286)
- @Vlad-Shcherbina Fix not one but two leaks related to gradients (#296)
- @scttfrdmn Fix: Add missing Float64 pattern in safetensors conversion (#295)
- @Vlad-Shcherbina Add missing #[param] attributes to InstanceNorm (#300)

## 0.25.2

- Introduce initial support for mlx-lm
  - impl `Parameter` trait for `Option<T>` where `T: ModuleParameters`
  - Add `finfo_max` and `finfo_min`
  - impl `Quantizable` for `Option<T>` where `T: Quantizable`

## 0.25.1

- Fix bug with `index_mut`

## 0.25.0

- Update `mlx-c` to version "0.2.0" and changes function signatures to
  match the new API
- Update `thiserror` to version "2"
- Fix wrong states number in `compile_with_state`
- Remove unnecessary evaluation in fft ops

## 0.23.0

- Update `mlx-c` to "0.1.2"
- Added `dilation` and `groups` parameters to the convolution layer

## 0.21.1

- Fix `mlx-sys` dependency to patch version in workspace

## 0.21.0

- Initial feature-complete release
