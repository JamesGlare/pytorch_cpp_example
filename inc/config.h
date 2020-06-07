#ifndef __RUNCONFIG__
#define __RUNCONFIG__
#include <stdint.h> // defines uin32_t, ...

namespace config{
    struct TrainConfig {
        uint32_t n_batch;
        uint32_t n_sample;
        float lr;
    };
    struct EvalConfig {
        uint32_t n_batch;
        uint32_t n_sample;
    };
}
#endif