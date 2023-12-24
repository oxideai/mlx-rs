#pragma once

#include <sstream>

#include "mlx/mlx.h"

namespace extra {
    // autocxx isn't yet smart enough to do anything with the R2Point
    // structure, so here we've manually made a cheeky little API to
    // do something useful with it.
    inline std::string hello() {
        std::ostringstream oss;
        oss << "hello";
        return oss.str();
    }

    using namespace mlx::core;

    class Foo {
    private:
    void* ptr_;

    public:
    Foo(void* ptr) : ptr_(ptr){};

    // Get the raw data pointer from the buffer
    void* raw_ptr();

    // Get the buffer pointer from the buffer
    const void* ptr() const {
        return ptr_;
    };
    void* ptr() {
        return ptr_;
    };
    };

    using foo = std::function<void(Foo)>;

    std::string bar(foo f);

    // class array2 {
    //     public:
    //       /** Construct a scalar array with zero dimensions. */


    //     /** The size of the array's datatype in bytes. */
    //     size_t itemsize() const {
    //         return size_of(dtype());
    //     };

    //     /** The number of elements in the array. */
    //     size_t size() const {
    //         return array_desc_->size;
    //     };

    //     /** The number of bytes in the array. */
    //     size_t nbytes() const {
    //         return size() * itemsize();
    //     };

    //     /** The number of dimensions of the array. */
    //     size_t ndim() const {
    //         return array_desc_->shape.size();
    //     };

    //     /** The shape of the array as a vector of integers. */
    //     const std::vector<int>& shape() const {
    //         return array_desc_->shape;
    //     };

    //     /**
    //      *  Get the size of the corresponding dimension.
    //      *
    //      *  This function supports negative indexing and provides
    //      *  bounds checking. */
    //     int shape(int dim) const {
    //         return shape().at(dim < 0 ? dim + ndim() : dim);
    //     };

    //     /** The strides of the array. */
    //     const std::vector<size_t>& strides() const {
    //         return array_desc_->strides;
    //     };

    //     /** Get the arrays data type. */
    //     Dtype dtype() const {
    //         return array_desc_->dtype;
    //     };

    //     /** Evaluate the array. */
    //     void eval(bool retain_graph = false);

    //       /** Get the value from a scalar array. */
    //     template <typename T>
    //     T item(bool retain_graph = false);

    //     struct ArrayIterator {
    //         using iterator_category = std::random_access_iterator_tag;
    //         using difference_type = size_t;
    //         using value_type = const array;
    //         using reference = value_type;

    //         explicit ArrayIterator(const array2& arr, int idx = 0) : arr(arr), idx(idx) {
    //         if (arr.ndim() == 0) {
    //             throw std::invalid_argument("Cannot iterate over 0-d array.");
    //         }
    //         }

    //         reference operator*() const;

    //         ArrayIterator& operator+(difference_type diff) {
    //         idx += diff;
    //         return *this;
    //         }

    //         ArrayIterator& operator++() {
    //         idx++;
    //         return *this;
    //         }

    //         friend bool operator==(const ArrayIterator& a, const ArrayIterator& b) {
    //         return a.arr.id() == b.arr.id() && a.idx == b.idx;
    //         };
    //         friend bool operator!=(const ArrayIterator& a, const ArrayIterator& b) {
    //         return !(a == b);
    //         };

    //     private:
    //         const array2& arr;
    //         int idx;
    //     };

    //     ArrayIterator begin() const {
    //         return ArrayIterator(*this);
    //     }
    //     ArrayIterator end() const {
    //         return ArrayIterator(*this, shape(0));
    //     }


    //     array2(
    //         const std::vector<int>& shape,
    //         Dtype dtype,
    //         std::unique_ptr<Primitive> primitive,
    //         const std::vector<array>& inputs);

    //     /** A unique identifier for an array. */
    //     std::uintptr_t id() const {
    //         return reinterpret_cast<std::uintptr_t>(array_desc_.get());
    //     }

    //     struct Data {
    //         allocator::Buffer buffer;
    //         deleter_t d;
    //         Data(allocator::Buffer buffer, deleter_t d = allocator::free)
    //             : buffer(buffer), d(d){};
    //         // Not copyable
    //         Data(const Data& d) = delete;
    //         Data& operator=(const Data& d) = delete;
    //         ~Data() {
    //         d(buffer);
    //         }
    //     };

    //     struct Flags {
    //         // True if there are no gaps in the underlying data. Each item
    //         // in the underlying data buffer belongs to at least one index.
    //         bool contiguous : 1;

    //         bool row_contiguous : 1;
    //         bool col_contiguous : 1;
    //     };

    //     /** The array's primitive. */
    //     Primitive& primitive() const {
    //         return *(array_desc_->primitive);
    //     };

    //     /** Check if the array has an attached primitive or is a leaf node. */
    //     bool has_primitive() const {
    //         return array_desc_->primitive != nullptr;
    //     };

    //     /** The array's inputs. */
    //     const std::vector<array>& inputs() const {
    //         return array_desc_->inputs;
    //     };

    //     /** A non-const reference to the array's inputs so that they can be used to
    //      * edit the graph. */
    //     std::vector<array>& editable_inputs() {
    //         return array_desc_->inputs;
    //     }

    //     /** Detach the array from the graph. */
    //     void detach();

    //     /** Get the Flags bit-field. */
    //     const Flags& flags() const {
    //         return array_desc_->flags;
    //     };

    //     /** The size (in elements) of the underlying buffer the array points to. */
    //     size_t data_size() const {
    //         return array_desc_->data_size;
    //     };

    //     allocator::Buffer& buffer() {
    //         return array_desc_->data->buffer;
    //     };
    //     const allocator::Buffer& buffer() const {
    //         return array_desc_->data->buffer;
    //     };

    //     template <typename T>
    //     T* data() {
    //         return static_cast<T*>(array_desc_->data_ptr);
    //     };

    //     template <typename T>
    //     const T* data() const {
    //         return static_cast<T*>(array_desc_->data_ptr);
    //     };

    //     // Check if the array has been evaluated
    //     bool is_evaled() const {
    //         return array_desc_->data != nullptr;
    //     }

    //     // Mark the array as a tracer array (true) or not.
    //     void set_tracer(bool is_tracer) {
    //         array_desc_->is_tracer = is_tracer;
    //     }
    //     // Check if the array is a tracer array
    //     bool is_tracer() const {
    //         return array_desc_->is_tracer;
    //     }

    //     // TODO: deleter_t is a std::function will give error in autocxx
    //     // void set_data(allocator::Buffer buffer, deleter_t d = allocator::free);

    //     // void set_data(
    //     //     allocator::Buffer buffer,
    //     //     size_t data_size,
    //     //     std::vector<size_t> strides,
    //     //     Flags flags,
    //     //     deleter_t d = allocator::free);

    //     void copy_shared_buffer(
    //         const array2& other,
    //         const std::vector<size_t>& strides,
    //         Flags flags,
    //         size_t data_size,
    //         size_t offset = 0);

    //     void copy_shared_buffer(const array2& other);

    //     void overwrite_descriptor(const array2& other) {
    //         array_desc_ = other.array_desc_;
    //     }

    //     private:
    //     struct ArrayDesc {
    //         std::vector<int> shape;
    //         std::vector<size_t> strides;
    //         size_t size;
    //         Dtype dtype;
    //         std::unique_ptr<Primitive> primitive{nullptr};

    //         // Indicates an array is being used in a graph transform
    //         // and should not be detached from the graph
    //         bool is_tracer{false};

    //         // This is a shared pointer so that *different* arrays
    //         // can share the underlying data buffer.
    //         std::shared_ptr<Data> data{nullptr};

    //         // Properly offset data pointer
    //         void* data_ptr{nullptr};

    //         // The size in elements of the data buffer the array accesses
    //         // This can be different than the actual size of the array if it
    //         // has been broadcast or irregularly strided.
    //         size_t data_size;

    //         // Contains useful meta data about the array
    //         Flags flags;

    //         std::vector<array> inputs;

    //         explicit ArrayDesc(const std::vector<int>& shape, Dtype dtype);

    //         explicit ArrayDesc(
    //             const std::vector<int>& shape,
    //             Dtype dtype,
    //             std::unique_ptr<Primitive> primitive,
    //             const std::vector<array>& inputs);

    //         ~ArrayDesc();
    //     };
    //     std::shared_ptr<ArrayDesc> array_desc_{nullptr};
    // };

    // // template <typename T>
    // // array2::array2(T val, Dtype dtype /* = TypeToDtype<T>() */)
    // //     : array_desc_(std::make_shared<ArrayDesc>(std::vector<int>{}, dtype)) {
    // // init(&val);
    // // }
}
