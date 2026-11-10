
use std::{fmt::Debug, ops::AddAssign};

use rand::distr::uniform::SampleUniform;
pub use crate::ff::*;

type WeightMatrix<OutputType> = Vec<OutputType>;

static mut RANDOM: Random<f32> = Random::<f32> (-1f32, 1f32);

/// Pops global random
pub fn set_random(value: Random<f32>) -> Random<f32> {
	unsafe {
		let old: Random<f32> = Random(RANDOM.0, RANDOM.1);
		RANDOM = value;
		return old;
	}
}
/// generates random number
pub fn rand() -> f32 {
	unsafe {
		return Random::<_>(RANDOM.0, RANDOM.1).rand();
	}
}
pub trait LayerConfig {
	fn get_layer_config(&self) -> &[usize];
}

#[derive(Debug, Default, Clone, Copy)]
pub enum LossType {
	#[default]
	Square,
	CrossEntropy,
}
#[derive(Debug)]
pub struct NeuralNetwork {
	/// Indexation is next: [layer][out + in * outs]
	pub(crate) weightmats: Vec<WeightMatrix<f32>>,
	pub(crate) biases: Vec<Vec<f32>>,
	pub(crate) layerconfig: Vec<usize>,
	pub(crate) learnrate: f32,
	pub(crate) acts: Vec<Box<dyn Activation>>,
	pub(crate) deltas: Vec<Vec<f32>>,
	pub gradients: NeuralNetworkGradients,
}
impl Default for NeuralNetwork {
	fn default() -> Self {
		return NeuralNetwork {
			weightmats: vec![],
			biases: vec![],
			layerconfig: vec![],
			learnrate: 0f32, 
			acts: vec![], 
			deltas: vec![], 
			gradients: NeuralNetworkGradients::default() 
		}
	}
}
impl ToString for NeuralNetwork {
	fn to_string(&self) -> String {
		let mut msg = String::new();
		msg = format!("{msg}\ncfg: {0:?}", self.get_layer_config());
		msg = format!("{msg}\nweights: {0:?}", self.weightmats);
		msg = format!("{msg}\nbiases: {0:?}", self.biases);
		return format!("{msg}\nlearnrate {0:?}", self.learnrate);
	}
}
#[derive(Debug, Default)]
pub struct NeuralNetworkInternal {
	/// Indexation is next: [layer][out]
	layers: Vec<Vec<f32>>
}
#[derive(Debug)]
pub struct NeuralNetworkGradients {
	weightmats: Vec<WeightMatrix<f32>>,
	biases: Vec<Vec<f32>>,
	pub count: usize,
}
impl Default for NeuralNetworkGradients {
	fn default() -> Self {
		return NeuralNetworkGradients { weightmats: vec![], biases: vec![], count: 0 };
	}
}
pub trait Generator<T: Clone> {
	fn rand(&self) -> T;
}
#[derive(Debug, Clone, Copy)]
pub struct Random<T: Clone + Copy + PartialOrd + SampleUniform> (pub T, pub T);
impl Default for Random<f32> {
	fn default() -> Self {
		return Random::<f32>(0.0, 1.0);
	}
}
impl Default for Random<usize> {
	fn default() -> Self {
		return Random::<usize>(0, 1);
	}
}
impl<T: Clone + Copy + PartialOrd + SampleUniform> Generator<T> for Random<T> {
	fn rand(&self) -> T {
		return rand::random_range(self.0..self.1);
	}
}
impl NeuralNetwork {
	pub fn get_loss(&self) -> LossType {
		return self.acts.last().expect("Last layer is magically was empty").get_loss_type();
	}
	pub fn get_learnrate(&mut self) -> f32 {
		self.learnrate
	}
	pub fn set_learnrate(&mut self, learnrate: f32) {
		self.learnrate = learnrate;
	}
	pub fn zero_deltas(layerconfig: &Vec<usize>) -> Vec<Vec<f32>> {
		let mut deltas = vec![];
		for weightmatidx in 0..layerconfig.len() - 1 {
			deltas.push(vec![0.0f32;layerconfig[weightmatidx + 1]]);
		}
		return deltas;
	}
	pub fn zero_gradients(layerconfig: &Vec<usize>) -> NeuralNetworkGradients {
		let mut gradients = NeuralNetworkGradients {
			weightmats: vec![],
			biases: vec![],
			count: 0
		};
		for idx in 1..layerconfig.len() {
			let inputs = layerconfig[idx - 1];
			let outputs = layerconfig[idx];
			gradients.weightmats.push(vec![0f32;inputs * outputs]);
			gradients.biases.push(vec![0f32;outputs]);
		}
		return gradients;
	}
	fn generate_layers(layerconfig: &Vec<usize>) -> (Vec<WeightMatrix<f32>>, Vec<Vec<f32>>) {
		let mut weights = vec![];
		let mut biases = vec![];
		for layeridx in 0..(layerconfig.len()-1) {
			let inputs = layerconfig[layeridx/* 0 */];
			let outputs = layerconfig[layeridx + 1 /* 1 */];
			// println!("generate_layers: inputs: {}, outputs: {}", inputs, outputs);
			weights.insert(layeridx, Vec::with_capacity(inputs * outputs));
			for weightidx in 0..(inputs * outputs) {
				weights[layeridx].insert(weightidx, rand());
			}
			biases.push(vec![0.1f32;outputs]);
		}
		return (weights, biases);
	}
	pub fn new(layerconfig: &Vec<usize>, learnrate: f32, acts: Vec<Box<dyn Activation>>) -> NeuralNetwork {
		let (weights, biases) = NeuralNetwork::generate_layers(layerconfig);
		let network = NeuralNetwork {
			weightmats: weights,
			biases: biases,
			layerconfig: layerconfig.clone(),
			learnrate: learnrate,
			acts: acts,
			deltas: Self::zero_deltas(layerconfig),
			gradients: Self::zero_gradients(layerconfig)
		};
		return network;
	}
}
impl<> LossCost<f32> for Vec<f32> {
	fn loss_cost(&self, losstype: LossType, expected: &[f32]) -> f32 {
		let mut loss = 0.0f32;
		// let output = &self.layers.last().unwrap();
		for k in 0..self.len() {
			match losstype {
				LossType::Square => {
					let d = self[k] - expected[k];
					loss += 0.5 * d * d;
				}
				LossType::CrossEntropy => {
					loss -= expected[k] * (self[k] + 1e-7).ln();
				}
			}
		}
		loss
	}
}
impl FeedForward<NeuralNetworkInternal> for NeuralNetwork {
	fn forward_internal(&self, input: &[f32]) -> NeuralNetworkInternal {
		let mut network: NeuralNetworkInternal = NeuralNetworkInternal { layers: vec![] };
		let layer = &self.weightmats[0];
		let inputs = input.len();
		let outputs = self.get_layer_config()[1];
		let mut sums: Vec<f32> = self.biases[0].clone();
		for outidx in 0..outputs {
			for inidx in 0..inputs {
				sums[outidx].add_assign(layer[outidx + inidx * outputs] * input[inidx]);
			}
			// out[outidx] = self.acts.activate(sum);
		}
		network.layers.push(input.to_vec());
		network.layers.push(self.acts[0].activate(&sums));
		for layeridx in 1..self.get_layer_config().len()-1 {
			// out + in * outs
			let layer = &self.weightmats[layeridx];
			let inputs = self.get_layer_config()[layeridx];
			let outputs = self.get_layer_config()[layeridx + 1];
			sums = self.biases[layeridx].clone();
			for outidx in 0..outputs {
				// println!("forward_internal: {outidx}/{outputs}: {0}", sums.len());
				(0..inputs) 
					.for_each(|inidx|
						sums[outidx].add_assign(layer[outidx + inidx * outputs] * network.layers[layeridx][inidx])
					);
			}
			network.layers.push(self.acts[layeridx].activate(&sums));
			// println!("forward_internal: network.layers.len(): {}", network.layers.len());
		}
		network
	}

	fn forward(&self, input: &[f32]) -> Vec<f32> {
		let mut last = input.to_vec().clone();
		for layeridx in 0..self.get_layer_config().len()-1 {
			// in + out * ins
			let layer = &self.weightmats[layeridx];
			let inputs = self.get_layer_config()[layeridx];
			let outputs = self.get_layer_config()[layeridx + 1];
			let mut sums: Vec<f32> = self.biases[layeridx].clone();
			for outidx in 0..outputs {
				(0..inputs).for_each(|inidx| sums[outidx].add_assign(layer[outidx + inidx * outputs] * last[inidx]));
				// for inidx in 0..inputs {
				// 	sums[outidx].add_assign(layer[outidx + inidx * outputs] * last[inidx]);
				// }
			}
			last = self.acts[layeridx].activate(&sums);
		}
		return last;
	}
}

// impl NeuralNetwork {
// 	fn partialbackprop(&mut self, weights: &Vec<f32>, inputs: usize, outputs: usize) {}
// }

impl BackPropogation<NeuralNetworkInternal> for NeuralNetwork {
	fn backpropogation(&mut self, input: &[f32], expected: &[f32]) -> Vec<f32> {
		let internal = self.forward_internal(input);
		let weightmatcount = self.weightmats.len();
		let mut derivs: Vec<f32>;
		// #1 Last layer
		let weightmatlast = weightmatcount - 1;
		let acts = self.acts[weightmatlast].activate(&internal.layers[weightmatcount]);
		if !self.acts[weightmatlast].vector_derivate(&acts, expected, &mut self.deltas[weightmatlast]) {
			derivs = vec![0f32;internal.layers[weightmatlast].len()];
			// println!("acts: {} == int: {} == deriv: {} == expected: {}", acts.len(), internal.layers[weightmatlast].len(), derivs.len(), expected.len());
			// println!("internal.layers[{weightmatcount}].len(): {}", internal.layers[weightmatcount].len());
			self.acts[weightmatlast].derivate( &internal.layers[weightmatcount], &mut derivs);
			for outidx in 0..internal.layers[weightmatcount].len() {
				let error = internal.layers[weightmatcount][outidx] - expected[outidx];
				match self.acts[weightmatlast].get_loss_type() {
					LossType::Square => self.deltas[weightmatlast][outidx] = error * derivs[outidx],
					LossType::CrossEntropy => self.deltas[weightmatlast][outidx] = error,
				}
			}
		}
		// #2 Hidden layers
		for layeridx in (0..weightmatcount - 1).rev() {
			let weights = &self.weightmats[layeridx + 1];
			let inputs = self.get_layer_config()[layeridx + 1]; 
			let outputs = self.get_layer_config()[layeridx + 2];
			derivs = vec![0f32;internal.layers[layeridx + 1].len()];
			self.acts[layeridx].derivate(&internal.layers[layeridx + 1], &mut derivs);
			for inidx in 0..inputs {
				let mut sum = 0.0f32;
				for outidx in 0..outputs {
					sum += weights[outidx + inidx * outputs] * self.deltas[layeridx+1][outidx];
				}
				self.deltas[layeridx][inidx] = sum * derivs[inidx];
			}
		}
		// #3 update gradient
		for weightmatidx in 0..weightmatcount {
			let inputs = self.get_layer_config()[weightmatidx];
			let outputs = self.get_layer_config()[weightmatidx + 1];
			let weights = &mut self.gradients.weightmats[weightmatidx];
			let biases = &mut self.gradients.biases[weightmatidx];
			let last_values = &internal.layers[weightmatidx];
			let current_deltas = &self.deltas[weightmatidx];

			for outidx in 0..outputs {
				for inidx in 0..inputs {
					let gradient = current_deltas[outidx] * last_values[inidx];
					weights[outidx + inidx * outputs] += gradient;
				}
				biases[outidx] += current_deltas[outidx];
			}
		}
		self.gradients.count += 1;
		let Some(v) = internal.layers.last() else {
			return vec![];
		};
		return v.clone();
	}
	
	fn update(&mut self) {
		let max = self.gradients.count.max(1) as f32;
		for weightmatidx in 0..self.weightmats.len() {
			let weights = &mut self.weightmats[weightmatidx];
			let biases = &mut self.biases[weightmatidx];
			let gradient_weights = &mut self.gradients.weightmats[weightmatidx];
			let gradient_biases = &mut self.gradients.biases[weightmatidx];
			for idx in 0..weights.len() {
				weights[idx] -= self.learnrate * (gradient_weights[idx] / max);
				gradient_weights[idx] = 0f32;
			}
			for idx in 0..biases.len() {
				biases[idx] -= self.learnrate * (gradient_biases[idx] / max);
				gradient_biases[idx] = 0f32;
			}
		}
		self.gradients.count = 0;
	}
}
impl LayerConfig for NeuralNetwork {
	fn get_layer_config(&self) -> &[usize] {
		return &self.layerconfig;
	}
}

