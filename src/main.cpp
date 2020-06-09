#include <iostream>
#include <thread>
#include <torch/torch.h>
#include "models.h"
#include "utils.h"
/* Three things to set up here. 
 * 1. Cmake (quick start + CMakeLists.txt)
 * 2. Debugger (gdb, launch.json)
 * 3. Intellisense (c_cpp_properties -> two include paths for pytorch)
 */ 

// If you *want* to see assembly
// just go make src/main.s
// Create your own tasks in vscode
// https://code.visualstudio.com/Docs/editor/tasks

// For debugging, use the gbp + vscode
// need to have compiled in debug mode
// make sure you've checked up launch.json

auto main(int, char**) -> int {
    
    return EXIT_SUCCESS;
}
