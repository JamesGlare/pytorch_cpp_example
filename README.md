# Minimal Torch-1.5-C++ Frontend Example

This repo contains a minimal cmake-based torch-C++ compile environment for Windows and Linux.
The repo relies on the awesome CLIUtils/cmake repository for including googletest for unit testing.
There is a pretty good tutorial for a minimal torch C++ build on `https://pytorch.org/cppdocs/installing.html`.


## Setup
1. Make sure you have `cmake>=3.1` and `g++ >= 5` or a new version of `msvc` installed.

2. Change the `CMAKE_PREFIX_PATH` in  CMakeLists.txt to the absolute path of libtorch you downloaded 
Linux -> `wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip && unzip libtorch-shared-with-deps-latest.zip`
Windows -> download `https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.5.0.zip`

3. Create a build directory (`mkdir build`) and `cd` in there. Then type `cmake -DCMAKE_BUILD_TYPE=DEBUG ..`.
If you want a build with optimizations, change build type to `RELEASE`.

4. In the same folder, type `cmake --build .`.

Note: If you use vscode, you can use the shortcuts provided in the `.vscode/tasks.json`, by typing Ctrl+Shift+P and `Tasks:Run Tasks`->`Configure Debug` etc.
For C++ intellisense, change the `includepath` in `.vscode/c_cpp_properties.json` to libtorch folder.
