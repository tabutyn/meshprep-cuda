// SPDX-License-Identifier: MIT
#pragma once

#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#define PARALLEL_MATER_APP_HAS_NVTX3 1
#else
#define PARALLEL_MATER_APP_HAS_NVTX3 0
#endif

namespace waterlab::detail {

class NvtxRange {
  public:
    explicit NvtxRange(const char *name)
#if PARALLEL_MATER_APP_HAS_NVTX3
        : range_(name)
#endif
    {
#if !PARALLEL_MATER_APP_HAS_NVTX3
        (void)name;
#endif
    }

  private:
#if PARALLEL_MATER_APP_HAS_NVTX3
    nvtx3::scoped_range range_;
#endif
};

} // namespace waterlab::detail

#undef PARALLEL_MATER_APP_HAS_NVTX3
