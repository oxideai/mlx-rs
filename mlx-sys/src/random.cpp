#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/array.h"

#include "mlx-cxx/random.hpp"
#include "mlx-cxx/mlx_cxx.hpp"

namespace mlx_cxx
{
    // TODO: is it possible to return a reference?
    std::optional<mlx::core::array> to_std_optional(const mlx_cxx::OptionalArray &opt)
    {
        switch (opt.tag)
        {
        case mlx_cxx::OptionalArray::Tag::None:
            return std::nullopt;
        case mlx_cxx::OptionalArray::Tag::Some:
            return *opt.payload.some;
        }
    }

    std::unique_ptr<mlx::core::array> key(uint64_t seed)
    {
        auto array = mlx::core::random::key(seed);
        return std::make_unique<mlx::core::array>(array);
    }

    // TODO: mlx::core::random::seed doesn't need wrapping

    std::unique_ptr<mlx::core::array> bits(
        const std::vector<int> &shape,
        int width,
        const mlx_cxx::OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bits(shape, width, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bits(
        const std::vector<int> &shape,
        const mlx_cxx::OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bits(shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::array<std::unique_ptr<mlx::core::array>, 2> split(
        const mlx::core::array &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto [key1, key2] = mlx::core::random::split(key, s.to_variant());
        return {
            std::make_unique<mlx::core::array>(key1),
            std::make_unique<mlx::core::array>(key2)};
    }

    std::unique_ptr<mlx::core::array> split(
        const mlx::core::array &key,
        int num,
        mlx_cxx::StreamOrDevice s)
    {
        auto array = mlx::core::random::split(key, num, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

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
        const mlx::core::array &low,
        const mlx::core::array &high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_bool(
        bool low,
        bool high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_uint8(
        uint8_t low,
        uint8_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_uint16(
        uint16_t low,
        uint16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_uint32(
        uint32_t low,
        uint32_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_uint64(
        uint64_t low,
        uint64_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_int8(
        int8_t low,
        int8_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_int16(
        int16_t low,
        int16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_int32(
        int32_t low,
        int32_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_int64(
        int64_t low,
        int64_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_float16(
        mlx::core::float16_t low,
        mlx::core::float16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_bfloat16(
        mlx::core::bfloat16_t low,
        mlx::core::bfloat16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_float32(
        float low,
        float high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> uniform_complex64(
        mlx::core::complex64_t low,
        mlx::core::complex64_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::uniform(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> normal(
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::normal(shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    // Default to float32
    std::unique_ptr<mlx::core::array> normal(
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::normal(shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint(
        const mlx::core::array &low,
        const mlx::core::array &high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_bool(
        bool low,
        bool high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_uint8(
        uint8_t low,
        uint8_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_uint16(
        uint16_t low,
        uint16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_uint32(
        uint32_t low,
        uint32_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_uint64(
        uint64_t low,
        uint64_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_int8(
        int8_t low,
        int8_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_int16(
        int16_t low,
        int16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_int32(
        int32_t low,
        int32_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_int64(
        int64_t low,
        int64_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_float16(
        mlx::core::float16_t low,
        mlx::core::float16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_bfloat16(
        mlx::core::bfloat16_t low,
        mlx::core::bfloat16_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_float32(
        float low,
        float high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> randint_complex64(
        mlx::core::complex64_t low,
        mlx::core::complex64_t high,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::randint(low, high, shape, dtype, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli(
        const mlx::core::array &p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli(
        const mlx::core::array &p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_bool(
        bool p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint8(
        uint8_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint16(
        uint16_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint32(
        uint32_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint64(
        uint64_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int8(
        int8_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int16(
        int16_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int32(
        int32_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int64(
        int64_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_float16(
        mlx::core::float16_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_bfloat16(
        mlx::core::bfloat16_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_float32(
        float p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_complex64(
        mlx::core::complex64_t p,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_bool(
        bool p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint8(
        uint8_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint16(
        uint16_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint32(
        uint32_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_uint64(
        uint64_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int8(
        int8_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int16(
        int16_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int32(
        int32_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_int64(
        int64_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_float16(
        mlx::core::float16_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_bfloat16(
        mlx::core::bfloat16_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_float32(
        float p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli_complex64(
        mlx::core::complex64_t p,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(p, shape, key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> bernoulli(
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice s)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::bernoulli(key_std, s.to_variant());
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> truncated_normal(
        const mlx::core::array &lower,
        const mlx::core::array &upper,
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::truncated_normal(lower, upper, shape, dtype, key_std);
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> truncated_normal(
        const mlx::core::array &lower,
        const mlx::core::array &upper,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::truncated_normal(lower, upper, dtype, key_std);
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> gumbel(
        const std::vector<int> &shape,
        mlx::core::Dtype dtype,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::gumbel(shape, dtype, key_std);
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> categorical(
        const mlx::core::array &logits,
        int axis,
        const std::vector<int> &shape,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::categorical(logits, axis, shape, key_std);
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> categorical(
        const mlx::core::array &logits,
        int axis,
        int num_samples,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::categorical(logits, axis, num_samples, key_std);
        return std::make_unique<mlx::core::array>(array);
    }

    std::unique_ptr<mlx::core::array> categorical(
        const mlx::core::array &logits,
        int axis,
        const OptionalArray &key,
        mlx_cxx::StreamOrDevice)
    {
        auto key_std = mlx_cxx::to_std_optional(key);
        auto array = mlx::core::random::categorical(logits, axis, key_std);
        return std::make_unique<mlx::core::array>(array);
    }
}