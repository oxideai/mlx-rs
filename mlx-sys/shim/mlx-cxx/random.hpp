#pragma once

#include "mlx/random.h"
#include "mlx/ops.h"
#include "mlx/array.h"

#include "mlx-cxx/mlx_cxx.hpp"
#include "mlx-cxx/utils.hpp"

namespace mlx_cxx {
    using OptionalArray = mlx_cxx::Optional<std::unique_ptr<mlx::core::array>>;

    std::optional<mlx::core::array> to_std_optional(const OptionalArray &opt);

    /// @brief Constructor binding for KeySequence
    /// @param seed 
    /// @return 
    std::unique_ptr<mlx::core::random::KeySequence> key_sequence(uint64_t seed);

    const mlx::core::random::KeySequence& default_key_sequence() {
        return mlx::core::random::KeySequence::default_();
    }

    /// @brief Get a PRNG key from a seed.
    /// @param seed 
    /// @return 
    std::unique_ptr<mlx::core::array> key(uint64_t seed);

    /// @brief Generate an array with type uint32 filled with random bits.
    /// @param shape 
    /// @param width 
    /// @param key 
    /// @param s 
    /// @return 
    std::unique_ptr<mlx::core::array> bits(
        const std::vector<int> &shape,
        int width,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s = {});

    /// @brief Generate an array with type uint32 filled with random bits.
    /// @param shape
    /// @param key
    /// @param s
    std::unique_ptr<mlx::core::array> bits(
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s = {});

    /// @brief Split the rng key into a pair of keys.
    /// @param key
    /// @param s
    /// @return
    std::array<std::unique_ptr<mlx::core::array>, 2> split(
        const mlx::core::array &key,
        mlx_cxx::StreamOrDevice s = {});

    /// @brief Split the rng key into `num` keys.
    /// @param key 
    /// @param num 
    /// @param s 
    /// @return 
    std::unique_ptr<mlx::core::array> split(
        const mlx::core::array &key,
        int num,
        mlx_cxx::StreamOrDevice s = {});

    // enum class Val {
    //     bool_,
    //     uint8,
    //     uint16,
    //     uint32,
    //     uint64,
    //     int8,
    //     int16,
    //     int32,
    //     int64,
    //     float16,
    //     float32,
    //     bfloat16,
    //     complex64,
    //   };

    std::unique_ptr<mlx::core::array> uniform(
        const mlx::core::array& low,
        const mlx::core::array& high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    /// TODO: template <typename T, typename U>
    std::unique_ptr<mlx::core::array> uniform_bool(
        bool low,
        bool high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_uint8(
        uint8_t low,
        uint8_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_uint16(
        uint16_t low,
        uint16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_uint32(
        uint32_t low,
        uint32_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_uint64(
        uint64_t low,
        uint64_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_int8(
        int8_t low,
        int8_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_int16(
        int16_t low,
        int16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_int32(
        int32_t low,
        int32_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_int64(
        int64_t low,
        int64_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_float16(
        mlx::core::float16_t low,
        mlx::core::float16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> uniform_bfloat16(
        mlx::core::bfloat16_t low,
        mlx::core::bfloat16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});
    
    std::unique_ptr<mlx::core::array> uniform_float32(
        float low,
        float high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});
    
    std::unique_ptr<mlx::core::array> uniform_complex64(
        mlx::core::complex64_t low,
        mlx::core::complex64_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    // std::unique_ptr<mlx::core::array> normal(
    //     const std::vector<int>& shape,
    //     mlx::core::Dtype dtype,
    //     const OptionalArray& key,
    //     mlx_cxx::StreamOrDevice s = {});

    // // Default to float32
    // std::unique_ptr<mlx::core::array> normal(
    //     const std::vector<int>& shape,
    //     const OptionalArray& key,
    //     mlx_cxx::StreamOrDevice s = {});

    /** Generate samples from the standard normal distribution. */
    std::unique_ptr<mlx::core::array> normal(
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const float loc,
        const float scale,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s);

    // // TODO: generate bindings for these overloads?
    // inline std::unique_ptr<mlx::core::array> normal(
    //     const std::vector<int>& shape,
    //     const float loc,
    //     const float scale,
    //     const std::optional<mlx::core::array>& key = std::nullopt,
    //     StreamOrDevice s = {}) {
    // return normal(shape, mlx::core::float32, loc, scale, key, s);
    // }
    // inline std::unique_ptr<mlx::core::array> normal(
    //     const std::vector<int>& shape,
    //     const mlx::core::Dtype dtype,
    //     const std::optional<mlx::core::array>& key = std::nullopt,
    //     StreamOrDevice s = {}) {
    // return normal(shape, dtype, 0.0, 1.0, key, s);
    // }
    // inline std::unique_ptr<mlx::core::array> normal(
    //     const std::vector<int>& shape,
    //     const std::optional<mlx::core::array>& key = std::nullopt,
    //     StreamOrDevice s = {}) {
    // return normal(shape, mlx::core::float32, 0.0, 1.0, key, s);
    // }

    std::unique_ptr<mlx::core::array> multivariate_normal(
        const mlx::core::array& mean,
        const mlx::core::array& cov,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        StreamOrDevice s = {}
    );

    std::unique_ptr<mlx::core::array> randint(
        const mlx::core::array& low,
        const mlx::core::array& high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> randint_bool(
        bool low,
        bool high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> randint_uint8(
        uint8_t low,
        uint8_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_uint16(
        uint16_t low,
        uint16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_uint32(
        uint32_t low,
        uint32_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_uint64(
        uint64_t low,
        uint64_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_int8(
        int8_t low,
        int8_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_int16(
        int16_t low,
        int16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_int32(
        int32_t low,
        int32_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_int64(
        int64_t low,
        int64_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_float16(
        mlx::core::float16_t low,
        mlx::core::float16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_bfloat16(
        mlx::core::bfloat16_t low,
        mlx::core::bfloat16_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_float32(
        float low,
        float high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> randint_complex64(
        mlx::core::complex64_t low,
        mlx::core::complex64_t high,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice = {});

    std::unique_ptr<mlx::core::array> bernoulli(
        const mlx::core::array& p,
        const std::vector<int>& shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli(
        const mlx::core::array& p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_bool(
        bool p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint8(
        uint8_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint16(
        uint16_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint32(
        uint32_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint64(
        uint64_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int8(
        int8_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int16(
        int16_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int32(
        int32_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int64(
        int64_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_float16(
        mlx::core::float16_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_bfloat16(
        mlx::core::bfloat16_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_float32(
        float p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_complex64(
        mlx::core::complex64_t p,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});


    std::unique_ptr<mlx::core::array> bernoulli_bool(
        bool p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint8(
        uint8_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint16(
        uint16_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint32(
        uint32_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_uint64(
        uint64_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int8(
        int8_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int16(
        int16_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int32(
        int32_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_int64(
        int64_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_float16(
        mlx::core::float16_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_bfloat16(
        mlx::core::bfloat16_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_float32(
        float p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> bernoulli_complex64(
        mlx::core::complex64_t p,
        const std::vector<int> &shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    // TODO: ignore this for now
    std::unique_ptr<mlx::core::array> bernoulli(
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice s = {});

    std::unique_ptr<mlx::core::array> truncated_normal(
        const mlx::core::array& lower,
        const mlx::core::array& upper,
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice);

    std::unique_ptr<mlx::core::array> truncated_normal(
        const mlx::core::array& lower,
        const mlx::core::array& upper,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice);

    std::unique_ptr<mlx::core::array> gumbel(
        const std::vector<int>& shape,
        mlx::core::Dtype dtype,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice);

    std::unique_ptr<mlx::core::array> categorical(
        const mlx::core::array& logits,
        int axis,
        const std::vector<int>& shape,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice);

    std::unique_ptr<mlx::core::array> categorical(
        const mlx::core::array& logits,
        int axis,
        int num_samples,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice);

    std::unique_ptr<mlx::core::array> categorical(
        const mlx::core::array& logits,
        int axis,
        const OptionalArray& key,
        mlx_cxx::StreamOrDevice);
}