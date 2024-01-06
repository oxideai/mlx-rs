#include "mlx-cxx/array.hpp"
#include <iostream>

namespace mlx_cxx {
    // TODO: remove this later. This is just a test for linking.
    void hello() {
        std::cout << "Hello, World!" << std::endl;
    }

    std::unique_ptr<array> array_new_bool(bool value) {
        return std::make_unique<array>(array(value));
    }
}
