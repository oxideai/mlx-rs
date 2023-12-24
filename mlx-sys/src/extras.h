#pragma once

#include <sstream>

namespace extra {
    // autocxx isn't yet smart enough to do anything with the R2Point
    // structure, so here we've manually made a cheeky little API to
    // do something useful with it.
    inline std::string hello() {
        std::ostringstream oss;
        oss << "hello";
        return oss.str();
    }
}
