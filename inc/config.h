#ifndef __RUNCONFIG__
#define __RUNCONFIG__
#include <stdint.h> // defines uin32_t, ...

namespace config{
    struct TrainConfig {
        uint32_t n_batch;
        uint32_t n_sample;
        float lr;
    };
    struct TestConfig {
        uint32_t n_batch;
        uint32_t n_sample;
    };
    struct EvolConfig {
        uint32_t max_layers;
        uint32_t min_layer_size;
        uint32_t max_layer_size;
        uint32_t max_rounds;
        uint32_t population;
        double pchange;
        double std;
        double noise_std;
    };
}
#endif