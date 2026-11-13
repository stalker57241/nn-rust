pub mod core {
    use rand::seq::IndexedRandom;

    pub struct Transition {
        pub state: Vec<f32>,      // Текущий кадр экрана
        pub action: usize,        // Нажатая кнопка
        pub reward: f32,          // Полученная награда от Си
        pub next_state: Vec<f32>, // Следующий кадр экрана
        pub done: bool,           // Флаг конца игры (Game Over)
    }

    pub struct ReplayBuffer {
        pub buffer: Vec<Transition>,
        pub capacity: usize,
        pub position: usize,
    }
    impl ReplayBuffer {
        pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
            let mut rng = rand::rng();
            let count = batch_size.min(self.buffer.len());
            return self.buffer.sample(&mut rng, count).collect();
        }
        pub fn len(&self) -> usize {
            return self.buffer.len();
        }
    }

    pub trait Syncable: Clone {
        fn sync_from(&mut self, other: &Self);
    }
    pub trait QModel: Syncable {
        fn predict(&self, state: &[f32]) -> Vec<f32>;
        fn train(&mut self, batch: &[(Vec<f32>, Vec<f32>)]);
    }
    pub trait ExplorationPolicy {
        fn select_action(&mut self, q_values: &[f32]) -> usize;
        fn on_step(&mut self);
    }
    pub struct EpsilonGreedy {
        pub epsilon: f32,
        pub min_epsilon: f32,
        pub decay: f32
    }
}

pub mod implementation {

use crate::{ff::{BackPropogation, FeedForward}, neuro::{Generator, NeuralNetwork, NeuralNetworkGradients, Random}, rl::core::*};
    impl Syncable for NeuralNetwork {
        fn sync_from(&mut self, other: &Self) {
            for (dst, src) in self.weightmats.iter_mut().zip(other.weightmats.iter()) {
                dst.as_mut_slice().copy_from_slice(src.as_slice());
            }
            for (dst, src) in self.biases.iter_mut().zip(other.weightmats.iter()) {
                dst.copy_from_slice(src);
            }
        }
    }
    impl QModel for NeuralNetwork {
        fn predict(&self, state: &[f32]) -> Vec<f32> {
            return self.forward(state);
        }
        fn train(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) {
            let mut gradient: NeuralNetworkGradients = self.get_gradients();
            for (state, target) in batch {
                let (_, gradients) = self.backpropogation(state, target);
                gradient.add_assign(&gradients);
            }
            self.update(&mut gradient);
        }
    }
    impl ExplorationPolicy for EpsilonGreedy {
        fn select_action(&mut self, q_values: &[f32]) -> usize {
            let random = Random(0.0f32, 1.0f32);
            if random.rand() < self.epsilon {
                return Random(0usize, q_values.len()).rand();
            }
            return q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

        }
        fn on_step(&mut self) {
            self.epsilon = (self.epsilon * self.decay).max(self.min_epsilon);
        }

    }
    pub struct DQN<M: QModel, P: ExplorationPolicy> {
        pub network: M,
        pub target: M,
        pub policy: P,
        pub memory: ReplayBuffer,
        pub gamma: f32,
        pub steps: usize,
        pub sync_interval: usize
    }
    impl<M: QModel, P: ExplorationPolicy> DQN<M, P> {
        pub fn new(network: M, policy: P, capacity: usize) -> Self {
            let target_network = network.clone();
            return Self {
                network: network,
                target: target_network,
                policy: policy,
                memory: ReplayBuffer { buffer: Vec::with_capacity(capacity), capacity, position: 0 },
                gamma: 0.99,
                steps: 0,
                sync_interval: 1000
            };
        }
        pub fn act(&mut self, state: &[f32]) -> usize {
            let q_values = self.network.predict(state);
            let action = self.policy.select_action(&q_values);
            self.policy.on_step();
            return action;
        }
        pub fn train_step(&mut self, batch_size: usize) {
            if self.memory.buffer.len() < batch_size {
                return;
            }
            let transitions = self.memory.sample(batch_size);
            let mut training_batch = Vec::with_capacity(batch_size);
            for t in transitions {
                let mut target_q = self.network.predict(&t.state);
                let max_next_q = if t.done {
                    0.0f32
                } else {
                    self.target
                        .predict(&t.next_state)
                        .into_iter().fold(f32::NEG_INFINITY, f32::max)
                };
                target_q[t.action] = t.reward + self.gamma * max_next_q;
                training_batch.push((t.state.clone(), target_q));
            }
            self.network.train(&training_batch);
            self.steps += 1;
            if self.steps % self.sync_interval == 0 {
                self.target.sync_from(&self.network);
            }
        }
    }
}
pub mod capi {
    use std::{os::raw::c_float, ptr::null_mut};

use crate::{neuro::{LayerConfig, NeuralNetwork}, rl::{DQN, EpsilonGreedy}};

    #[unsafe(no_mangle)]
    pub extern "C" fn makePolicy(epsilon: c_float, min_epsilon: c_float, decay: c_float) -> *mut EpsilonGreedy {
        return Box::into_raw(Box::new(EpsilonGreedy { epsilon, min_epsilon, decay }));
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn dqnCreate(network: *mut NeuralNetwork, policy: *mut EpsilonGreedy, capacity: usize) -> *mut DQN<NeuralNetwork, EpsilonGreedy> {
        if network.is_null() || policy.is_null() {
            return null_mut();
        }
        let network = unsafe {network.read()};
        let policy = unsafe {policy.read()};
        let dqn = DQN::new(network, policy, capacity);
        return Box::into_raw(Box::new(dqn));
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn dqnAct(dqn: *mut DQN<NeuralNetwork, EpsilonGreedy>, state: *const c_float) -> usize {
        if dqn.is_null() || state.is_null() {
            return 0;
        }
        let state = unsafe { std::slice::from_raw_parts(state, dqn.read().network.get_layer_config()[0])};
        return unsafe { &mut *dqn }.act(state);
    }
    #[unsafe(no_mangle)]
    pub extern "C" fn dqnTrainStep(dqn: *mut DQN<NeuralNetwork, EpsilonGreedy>, batch_size: usize) {
        if dqn.is_null() {
            return;
        }
        unsafe { &mut *dqn }.train_step(batch_size);
    }
}

pub use self::core::*;
pub use self::implementation::*;