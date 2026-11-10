#ifndef _RS_NEURAL_H
#define _RS_NEURAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct NeuralNetwork NeuralNetwork;
enum ActivationKind {
    ACTIVATION_KIND_SIGMOID = 0,
    ACTIVATION_KIND_RELU = 1,
    ACTIVATION_KIND_LEAKYRELU = 2,
    ACTIVATION_KIND_SOFTMAX = 3,
    ACTIVATION_KIND_LINEAR = 4,
};

extern NeuralNetwork* nnCreate(
    const size_t* layers,
    const enum ActivationKind* acts,
    size_t layercount,
    float learnrate
);
extern int nnForward(
    NeuralNetwork* network,
    const float* input,
    float* poutput
);
extern int nnBackpropogate(
    NeuralNetwork* network,
    const float* input,
    const float* expected
);
extern void nnUpdate(NeuralNetwork* network);
extern void nnDestroy(NeuralNetwork* network);

extern int nnStore(const NeuralNetwork* network, char* filename);
extern NeuralNetwork* nnLoad(char* filename);

#endif /* _RS_NEURAL_H */