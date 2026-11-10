
pub mod helper;
pub mod rl;
pub mod ff;
pub mod neuro;
pub mod io;
pub mod capi;

pub mod minimal_required {
	pub use crate::{
		ff::{BackPropogation, LossCost, FeedForward},
		helper::{SIGMOID, SOFTMAX, one_hot},
		neuro::NeuralNetwork
	};
}
pub mod mnist_minimal_required {
    pub use crate::minimal_required::*;
    pub use crate::helper::io::load_images;
}