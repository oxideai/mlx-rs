/// Helper type alias and functions for the mlx opague types

#pragma once

#include <sstream>

#include "mlx/mlx.h"

namespace ext {
    inline std::string hello() {
        std::ostringstream oss;
        oss << "Hello, world!";
        return oss.str();
    }

    namespace array {
        using namespace mlx::core;

        /// @brief Type alias for mlx::core::array
        using MlxArray = mlx::core::array;
        using MlxArrayIterator = mlx::core::array::ArrayIterator;

        // There seems to have no support for templated functions in autocxx.
        // We need to manually write the functions for each type.
        // The type supported will be limited to variants of `Dtype`.
        // bool,
        // uint8,
        // uint16,
        // uint32,
        // uint64,
        // int8,
        // int16,
        // int32,
        // int64,
        // float16,
        // float32,
        // bfloat16,
        // complex64

        /* -------------------------------------------------------------------------- */
        /*                          Scalar array constructors                         */
        /* -------------------------------------------------------------------------- */
        
        /// @brief Construct a scalar array of boolean value with zero dimensions.
        /// @param val bool
        /// @return MlxArray
        MlxArray new_scalar_array_bool(bool val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of uint8 value with zero dimensions.
        /// @param val uint8_t
        /// @return MlxArray
        MlxArray new_scalar_array_uint8(uint8_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of uint16 value with zero dimensions.
        /// @param val uint16_t
        /// @return MlxArray
        MlxArray new_scalar_array_uint16(uint16_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of uint32 value with zero dimensions.
        /// @param val uint32_t
        /// @return MlxArray
        MlxArray new_scalar_array_uint32(uint32_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of uint64 value with zero dimensions.
        /// @param val uint64_t
        /// @return MlxArray
        MlxArray new_scalar_array_uint64(uint64_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of int8 value with zero dimensions.
        /// @param val int8_t
        /// @return MlxArray
        MlxArray new_scalar_array_int8(int8_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of int16 value with zero dimensions.
        /// @param val int16_t
        /// @return MlxArray
        MlxArray new_scalar_array_int16(int16_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of int32 value with zero dimensions.
        /// @param val int32_t
        /// @return MlxArray
        MlxArray new_scalar_array_int32(int32_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of int64 value with zero dimensions.
        /// @param val int64_t
        /// @return MlxArray
        MlxArray new_scalar_array_int64(int64_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of float16 value with zero dimensions.
        /// @param val float16_t
        /// @return MlxArray
        MlxArray new_scalar_array_float16(float16_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of float32 value with zero dimensions.
        /// @param val float
        /// @return MlxArray
        MlxArray new_scalar_array_float32(float val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of bfloat16 value with zero dimensions.
        /// @param val bfloat16_t
        /// @return MlxArray
        MlxArray new_scalar_array_bfloat16(bfloat16_t val) {
            return mlx::core::array(val);
        }

        /// @brief Construct a scalar array of complex64 value with zero dimensions.
        /// @param val complex64_t
        /// @return MlxArray
        MlxArray new_scalar_array_complex64(complex64_t val) {
            return mlx::core::array(val);
        }

        /* -------------------------------------------------------------------------- */

        // TODO: what about the iterator constructor?

        /* -------------------------------------------------------------------------- */

        /* -------------------------------------------------------------------------- */
        /*                        Initializer list constructors                       */
        /* -------------------------------------------------------------------------- */
        
        /// @brief Construct an array of boolean values from an initializer list.
        /// @param list std::initializer_list<bool>
        /// @return MlxArray
        MlxArray new_array_from_list_bool(std::initializer_list<bool> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of uint8 values from an initializer list.
        /// @param list std::initializer_list<uint8_t>
        /// @return MlxArray
        MlxArray new_array_from_list_uint8(std::initializer_list<uint8_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of uint16 values from an initializer list.
        /// @param list std::initializer_list<uint16_t>
        /// @return MlxArray
        MlxArray new_array_from_list_uint16(std::initializer_list<uint16_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of uint32 values from an initializer list.
        /// @param list std::initializer_list<uint32_t>
        /// @return MlxArray
        MlxArray new_array_from_list_uint32(std::initializer_list<uint32_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of uint64 values from an initializer list.
        /// @param list std::initializer_list<uint64_t>
        /// @return MlxArray
        MlxArray new_array_from_list_uint64(std::initializer_list<uint64_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of int8 values from an initializer list.
        /// @param list std::initializer_list<int8_t>
        /// @return MlxArray
        MlxArray new_array_from_list_int8(std::initializer_list<int8_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of int16 values from an initializer list.
        /// @param list std::initializer_list<int16_t>
        /// @return MlxArray
        MlxArray new_array_from_list_int16(std::initializer_list<int16_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of int32 values from an initializer list.
        /// @param list std::initializer_list<int32_t>
        /// @return MlxArray
        MlxArray new_array_from_list_int32(std::initializer_list<int32_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of int64 values from an initializer list.
        /// @param list std::initializer_list<int64_t>
        /// @return MlxArray
        MlxArray new_array_from_list_int64(std::initializer_list<int64_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of float16 values from an initializer list.
        /// @param list std::initializer_list<float16_t>
        /// @return MlxArray
        MlxArray new_array_from_list_float16(std::initializer_list<float16_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of float32 values from an initializer list.
        /// @param list std::initializer_list<float>
        /// @return MlxArray
        MlxArray new_array_from_list_float32(std::initializer_list<float> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of bfloat16 values from an initializer list.
        /// @param list std::initializer_list<bfloat16_t>
        /// @return MlxArray
        MlxArray new_array_from_list_bfloat16(std::initializer_list<bfloat16_t> list) {
            return mlx::core::array(list);
        }

        /// @brief Construct an array of complex64 values from an initializer list.
        /// @param list std::initializer_list<complex64_t>
        /// @return MlxArray
        MlxArray new_array_from_list_complex64(std::initializer_list<complex64_t> list) {
            return mlx::core::array(list);
        }

        /* -------------------------------------------------------------------------- */

        /* -------------------------------------------------------------------------- */
        /*                             Buffer constructor                             */
        /* -------------------------------------------------------------------------- */

        MlxArray new_array_from_buffer(allocator::Buffer data, const std::vector<int32_t>& shape, Dtype dtype) {
            return mlx::core::array(data, shape, dtype);
        }

        /* -------------------------------------------------------------------------- */

        // TODO: Copy and move constructors?

        /* -------------------------------------------------------------------------- */
        /*                          Other functions for array                         */
        /* -------------------------------------------------------------------------- */

        // Naming convention:
        // The function name is the same as the corresponding function in the
        // `mlx::core::array` class. Overloaded functions are named with a numeric
        // suffix. Template functions are named with a type suffix. Function with default
        // arguments are treated as overloaded functions. If the type ends with a
        // numeric suffix, an underscore is added between the type and the overload suffix.

        /// @brief Get the size of the array's datatype in bytes.
        /// @param array 
        /// @return size of the array's datatype in bytes
        size_t itemize(const MlxArray& array) {
            return array.itemsize();
        }

        /// @brief Get the number of elements in the array.
        /// @param array
        /// @return number of elements in the array
        size_t size(const MlxArray& array) {
            return array.size();
        }

        /// @brief Get the number of bytes in the array.
        /// @param array
        /// @return number of bytes in the array
        size_t nbytes(const MlxArray& array) {
            return array.nbytes();
        }

        /// @brief Get the number of dimensions of the array.
        /// @param array
        /// @return number of dimensions of the array
        size_t ndim(const MlxArray& array) {
            return array.ndim();
        }

        /// @brief Get the shape of the array as a vector of integers.
        /// @param array
        /// @return shape of the array as a vector of integers
        const std::vector<int32_t>& shape(const MlxArray& array) {
            return array.shape();
        }

        /// @brief Get the size of the corresponding dimension.
        /// @param array
        /// @param dim
        /// @return size of the corresponding dimension
        int32_t shape1(const MlxArray& array, int32_t dim) {
            return array.shape(dim);
        }

        /// @brief Get the strides of the array
        /// @param array 
        /// @return strides of the array
        const std::vector<size_t>& strides(const MlxArray& array) {
            return array.strides();
        }

        /// @brief Evaluate the array
        /// @param array 
        void eval(MlxArray array) {
            array.eval();
        }

        /// @brief Evaluate the array
        /// @param array
        /// @param retain_graph
        void eval1(MlxArray array, bool retain_graph) {
            array.eval(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        bool item_bool(MlxArray array) {
            return array.item<bool>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        bool item_bool_1(MlxArray array, bool retain_graph) {
            return array.item<bool>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        uint8_t item_uint8(MlxArray array) {
            return array.item<uint8_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        uint8_t item_uint8_1(MlxArray array, bool retain_graph) {
            return array.item<uint8_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        uint16_t item_uint16(MlxArray array) {
            return array.item<uint16_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        uint16_t item_uint16_1(MlxArray array, bool retain_graph) {
            return array.item<uint16_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        uint32_t item_uint32(MlxArray array) {
            return array.item<uint32_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        uint32_t item_uint32_1(MlxArray array, bool retain_graph) {
            return array.item<uint32_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        uint64_t item_uint64(MlxArray array) {
            return array.item<uint64_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        uint64_t item_uint64_1(MlxArray array, bool retain_graph) {
            return array.item<uint64_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        int8_t item_int8(MlxArray array) {
            return array.item<int8_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        int8_t item_int8_1(MlxArray array, bool retain_graph) {
            return array.item<int8_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        int16_t item_int16(MlxArray array) {
            return array.item<int16_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        int16_t item_int16_1(MlxArray array, bool retain_graph) {
            return array.item<int16_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        int32_t item_int32(MlxArray array) {
            return array.item<int32_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        int32_t item_int32_1(MlxArray array, bool retain_graph) {
            return array.item<int32_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        int64_t item_int64(MlxArray array) {
            return array.item<int64_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        int64_t item_int64_1(MlxArray array, bool retain_graph) {
            return array.item<int64_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        float16_t item_float16(MlxArray array) {
            return array.item<float16_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        float16_t item_float16_1(MlxArray array, bool retain_graph) {
            return array.item<float16_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        float item_float32(MlxArray array) {
            return array.item<float>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        float item_float32_1(MlxArray array, bool retain_graph) {
            return array.item<float>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        bfloat16_t item_bfloat16(MlxArray array) {
            return array.item<bfloat16_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        bfloat16_t item_bfloat16_1(MlxArray array, bool retain_graph) {
            return array.item<bfloat16_t>(retain_graph);
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @return value from a scalar array
        complex64_t item_complex64(MlxArray array) {
            return array.item<complex64_t>();
        }

        /// @brief Get the value from a scalar array.
        /// @param array
        /// @param retain_graph
        /// @return value from a scalar array
        complex64_t item_complex64_1(MlxArray array, bool retain_graph) {
            return array.item<complex64_t>(retain_graph);
        }

        /* -------------------------------------------------------------------------- */
        /*      TODO: APIs that should are intended for use by the backend only.      */
        /* -------------------------------------------------------------------------- */

    }
}