# Provenance

Thomas Butyn is the sole author and copyright owner of the imported implementation used as the historical baseline. That source was recovered from personal repositories and may be relicensed under MIT.

The `legacy-baseline` Git tag preserves only the owned CUDA implementation and matching declaration header. The release branch removes the historical API and product naming, then rebuilds the implementation behind the public `meshprep` C++20 API.

The separate historical packaging repository is not an upstream dependency and must not be published as this project. Its binaries, bytecode, ctypes wrapper, Blender scripts, and game assets are excluded.
