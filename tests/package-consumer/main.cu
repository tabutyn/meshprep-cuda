// SPDX-License-Identifier: MIT
#include <meshprep/meshprep.hpp>

int main()
{
    meshprep::Workspace workspace;
    return workspace.capacity_bytes() == 0 ? 0 : 1;
}
