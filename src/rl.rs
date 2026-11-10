pub mod core {
    pub trait Environment: Default {
        type State;
        type Action;

        fn reset(&mut self) -> &[Self::State];
        fn step(&mut self, action: Self::Action) -> (&[Self::State], f32, bool);
        fn action_space(&self) -> usize;
    }
    pub trait Policy<S, A> {
        fn select_action(&mut self, state: &[S]) -> A;
    }
    pub trait ValueFunction<S> {
        fn value(&self, state: &[S]) -> f32;
    }
    pub trait QFunction<S> {
        fn q_values(&self, state: &[S]) -> Vec<f32>;
    }
    pub trait ReplayBuffer<S>: Default {
        fn push(&mut self, state: &[S]);
        fn sample(&self, batch_size: usize) -> Vec<S>;
        fn len(&self) -> usize;
        fn clear(&mut self);
    }
    #[derive(Debug, Clone, Default)]
    pub struct Transition<'a, S, A> {
        pub state: &'a [S],
        pub action: A,
        pub reward: f32,
        pub next_state: &'a [S],
        pub done: bool,
    }
    pub trait Agent<E: Environment> {
        fn act(&mut self, state: &E::State) -> E::Action;
        fn observe(&mut self, transition: Transition<E::State, E::Action>);
        fn train_step(&mut self);
        fn reset_episode(&mut self);
    }
    pub trait EpsilonSchedule {
        fn epsilon(&self, step: usize) -> f32;
    }

    pub trait TargetNetworkUpdate {
        fn update_target(&mut self);
    }
}

pub use self::core::*;