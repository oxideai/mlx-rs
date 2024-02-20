#include "mlx-cxx/fast.hpp"

namespace mlx_cxx::fast {
        std::unique_ptr<mlx::core::array> rope(
        const mlx::core::array& x,
        int dims,
        bool traditional,
        float base,
        float scale,
        int offset,
        mlx_cxx::StreamOrDevice s)
    {
        auto arr = mlx::core::fast::rope(
            x,
            dims,
            traditional,
            base,
            scale,
            offset,
            s.to_variant());
        return std::make_unique<mlx::core::array>(arr);
    }
}