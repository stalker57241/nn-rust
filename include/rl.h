#ifndef _NN_RL_H
#define _NN_RL_H

#include "neural.h"
#include <cstddef>
typedef struct DQN DQN;
typedef struct EpsilonGreedy EpsilonGreedy;

extern EpsilonGreedy* makePolicy(float epsilon, float min_epsilon, float decay);

extern DQN* dqnCreate(NeuralNetwork* network, EpsilonGreedy policy, size_t capacity);
extern size_t dqnAct(DQN* dqn, float* state);
extern void dqnTrainStep(DQN* dqn, size_t batch_size);

#endif /* _NN_RL_H */