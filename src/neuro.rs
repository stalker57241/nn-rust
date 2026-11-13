
use std::{fmt::Debug, ops::AddAssign, sync::Arc, thread};

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
	fn get_layer_config(&self) -> Vec<usize>;
}

#[derive(Debug, Default, Clone, Copy)]
pub enum LossType {
	#[default]
	Square,
	CrossEntropy,
}
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
	/// Indexation is next: [layer][out + in * outs]
	pub(crate) weightmats: Vec<WeightMatrix<f32>>,
	pub(crate) biases: Vec<Vec<f32>>,
	pub(crate) bias_modulation: Vec<f32>,
	pub(crate) layerconfig: Vec<usize>,
	pub(crate) learnrate: f32,
	pub(crate) acts: Arc<Vec<Box<dyn Activation>>>
}
impl Default for NeuralNetwork {
	fn default() -> Self {
		return NeuralNetwork {
			weightmats: vec![],
			biases: vec![],
			bias_modulation: vec![],
			layerconfig: vec![],
			learnrate: 0f32,
			acts: Arc::new(vec![])
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
#[derive(Debug, Clone)]
pub struct NeuralNetworkGradients {
	weightmats: Vec<WeightMatrix<f32>>,
	biases: Vec<Vec<f32>>,
	pub count: usize,
}
impl NeuralNetworkGradients {
	pub fn new(layerconfig: Vec<usize>) -> Self {

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
	pub fn add_assign(&mut self, rhs: &Self) {
		self.count += rhs.count;
		for i in 0..self.weightmats.len() {
			for j in 0..self.weightmats[i].len() {
				self.weightmats[i][j] += rhs.weightmats[i][j];
			}
			for j in 0..self.biases[i].len() {
				self.biases[i][j] += rhs.biases[i][j];
			}
		}
	}
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
pub trait GetLoss {
	fn get_loss(&self) -> LossType;
}
impl GetLoss for NeuralNetwork {
	fn get_loss(&self) -> LossType {
		return self.acts.last().expect("Last layer is magically was empty").get_loss_type().clone();
	}
}
impl NeuralNetwork {
	pub fn get_learnrate(&mut self) -> f32 {
		self.learnrate
	}
	pub fn set_learnrate(&mut self, learnrate: f32) {
		self.learnrate = learnrate;
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
	pub fn new(acts: Vec<Box<dyn Activation>>, layerconfig: &Vec<usize>, learnrate: f32) -> NeuralNetwork {
		let (weights, biases) = NeuralNetwork::generate_layers(layerconfig);
		let network = NeuralNetwork {
			weightmats: weights,
			biases: biases,
			bias_modulation: vec![1.0f32; layerconfig.len() - 1],
			layerconfig: layerconfig.clone(),
			learnrate: learnrate,
			acts: Arc::from(acts)
		};
		return network;
	}
	pub fn get_gradients(&self) -> NeuralNetworkGradients {
		return NeuralNetworkGradients::new(
			self.get_layer_config()
		);
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
		let mut modulation: f32 = self.bias_modulation[0];
		let mut sums: Vec<f32> = self.biases[0].iter().map(|x| x * modulation).collect();
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
			// let inputs = self.get_layer_config()[layeridx];
			let outputs = self.get_layer_config()[layeridx + 1];
			modulation = self.bias_modulation[layeridx];
			sums = self.biases[layeridx].iter().map(|x| x * modulation).collect();
			for outidx in 0..outputs {
				// println!("forward_internal: {outidx}/{outputs}: {0}", sums.len());
				sums
					.iter_mut()
					.enumerate()
					.for_each(|(inidx, value)|
						value.add_assign(layer[outidx + inidx * outputs] * network.layers[layeridx][inidx]));
				// (0..inputs).for_each(|inidx|sums[outidx].add_assign(layer[outidx + inidx * outputs] * network.layers[layeridx][inidx]));
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
			let outputs = self.get_layer_config()[layeridx + 1];
			let modulation: f32 = self.bias_modulation[layeridx];
			let mut sums: Vec<f32> = self.biases[layeridx].iter().map(|x| x * modulation).collect();
			for outidx in 0..outputs {
				sums
					.iter_mut()
					.enumerate()
					.for_each(|(inidx, sum)|
						sum.add_assign(layer[outidx + inidx * outputs] * last[inidx])
					);
				
			}
			last = self.acts[layeridx].activate(&sums);
		}
		return last;
	}
}

impl BackPropogation<NeuralNetworkInternal, NeuralNetworkGradients> for NeuralNetwork {
	fn backpropogation(&self, input: &[f32], expected: &[f32]) -> (Vec<f32>, NeuralNetworkGradients) {
		let mut gradients: NeuralNetworkGradients = NeuralNetworkGradients::new(self.get_layer_config());
		let mut deltas: Vec<Vec<f32>> = self.layerconfig[1..]
			.iter()
			.map(|&n| vec![0.0; n])
			.collect();
		let internal = self.forward_internal(input);
		let weightmatcount = self.weightmats.len();
		let mut derivs: Vec<f32>;
		// #1 Last layer
		let weightmatlast = weightmatcount - 1;
		let acts = self.acts[weightmatlast].activate(&internal.layers[weightmatcount]);
		if !self.acts[weightmatlast].vector_derivate(&acts, expected, &mut deltas[weightmatlast]) {
			derivs = vec![0f32;internal.layers[weightmatlast].len()];
			self.acts[weightmatlast].derivate( &internal.layers[weightmatcount], &mut derivs);
			for outidx in 0..internal.layers[weightmatcount].len() {
				let error = internal.layers[weightmatcount][outidx] - expected[outidx];
				match self.acts[weightmatlast].get_loss_type() {
					LossType::Square => deltas[weightmatlast][outidx] = error * derivs[outidx],
					LossType::CrossEntropy => deltas[weightmatlast][outidx] = error,
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
					sum += weights[outidx + inidx * outputs] * deltas[layeridx+1][outidx];
				}
				deltas[layeridx][inidx] = sum * derivs[inidx];
			}
		}
		// #3 update gradient
		for weightmatidx in 0..weightmatcount {
			let inputs = self.get_layer_config()[weightmatidx];
			let outputs = self.get_layer_config()[weightmatidx + 1];
			let weights = &mut gradients.weightmats[weightmatidx];
			let biases = &mut gradients.biases[weightmatidx];
			let last_values = &internal.layers[weightmatidx];
			let current_deltas = &deltas[weightmatidx];

			for outidx in 0..outputs {
				for inidx in 0..inputs {
					let gradient = current_deltas[outidx] * last_values[inidx];
					weights[outidx + inidx * outputs] += gradient;
				}
				biases[outidx] += current_deltas[outidx];
			}
		}
		gradients.count += 1;
		let Some(v) = internal.layers.last() else {
			return (vec![], gradients);
		};
		return (v.clone(), gradients);
	}
	
	fn update(&mut self, gradients: &mut NeuralNetworkGradients) {
		let max = gradients.count.max(1) as f32;
		for weightmatidx in 0..self.weightmats.len() {
			let weights = &mut self.weightmats[weightmatidx];
			let biases = &mut self.biases[weightmatidx];
			let gradient_weights = &mut gradients.weightmats[weightmatidx];
			let gradient_biases = &mut gradients.biases[weightmatidx];
			for idx in 0..weights.len() {
				weights[idx] -= self.learnrate * (gradient_weights[idx] / max);
			}
			for idx in 0..biases.len() {
				biases[idx] -= self.learnrate * (gradient_biases[idx] / max);
			}
		}
	}
}
impl LayerConfig for NeuralNetwork {
	fn get_layer_config(&self) -> Vec<usize> {
		return self.layerconfig.clone();
	}
}

