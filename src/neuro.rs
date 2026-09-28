#[derive(Debug)]
pub struct Act(pub fn(f32) -> f32, pub fn(f32) -> f32);
impl Act {
	pub fn activate(&self, x: f32) -> f32 {
		return self.0(x);
	}
	pub fn derivate(&self, x: f32) -> f32 {
		return self.1(x);
	}
}
pub trait FeedForwardRaw {}
pub trait FeedForwardInternal<T: FeedForwardRaw, DataType = f32> {
	fn forward_internal(&self, input: &[DataType]) -> T;
}
pub trait FeedForward<Raw: FeedForwardRaw, DataType = f32>: FeedForwardInternal<Raw> {
	fn forward(&self, input: &[DataType]) -> Vec<DataType>;
	fn backward(&mut self, input: &[DataType], expected: &[DataType]) -> f32;
}
type WeightMatrix<OutputType> = Vec<OutputType>;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
	/// Indexation is next: [layer][in + out * ins]
	weightmats: Vec<WeightMatrix<f32>>,
	biases: Vec<Vec<f32>>,
	layerconfig: Vec<usize>,
	learnrate: f32,
	acts: &'static Act
}
impl ToString for NeuralNetwork {
	fn to_string(&self) -> String {
		let mut msg = String::new();
		msg = format!("{msg}\ncfg: {0:?}", self.layerconfig);
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
pub trait Instance<T> {
	fn instance(&self) -> T;
	fn zero(&self) -> T;
}
impl Instance<Self> for f32 {
	fn instance(&self) -> Self {
		return self.clone();
	}
	fn zero(&self) -> Self {
		return 0.0f32;
	}
}
impl Instance<Self> for f64 {
	fn instance(&self) -> Self {
		return self.clone();
	}
	fn zero(&self) -> Self {
		return 0.0f64;
	}
}
pub trait Generator<T: Clone> {
	fn rand(&self) -> T;
}
#[derive(Debug)]
pub struct Randomf32 (pub f32, pub f32);
impl Default for Randomf32 {
	fn default() -> Self {
		return Randomf32(0.0, 1.0);
	}
}
impl<> Generator<f32> for Randomf32 {
	fn rand(&self) -> f32 {
		return rand::random_range(self.0..self.1);
	}
}
impl<> Instance<f32> for Randomf32 {
	fn instance(&self) -> f32 {
		return self.rand();
	}
	fn zero(&self) -> f32 {
		return 0.0f32;
	}
}
impl NeuralNetwork {
	pub fn degrade_learnrate(&mut self, epoch: usize, tau: f32, lr0: f32, lr_end: f32) {
		// self.learnrate *= 0.99;
		self.learnrate = lr_end + (lr0 - lr_end) * (-(epoch as f32) / tau).exp();
	}
	pub fn get_learnrate(&mut self) -> f32 {
		self.learnrate
	}
	pub fn set_learnrate(&mut self, learnrate: f32) {
		self.learnrate = learnrate;
	}
	fn generate_layers(default: &impl Instance<f32>, layerconfig: &Vec<usize>) -> (Vec<WeightMatrix<f32>>, Vec<Vec<f32>>) {
		let mut weights = vec![];
		let mut biases = vec![];
		for layeridx in 0..(layerconfig.len()-1) {
			weights.insert(layeridx, Vec::with_capacity(layerconfig[layeridx] * layerconfig[layeridx + 1]));
			for weightidx in 0..layerconfig[layeridx] * layerconfig[layeridx + 1] {
				weights[layeridx].insert(weightidx, default.instance());
			}
			biases.push(vec![default.zero();layerconfig[layeridx + 1]]);
		}
		return (weights, biases);
	}
	pub fn new(default: impl Instance<f32>, layerconfig: &Vec<usize>, learnrate: f32, act: &'static Act) -> NeuralNetwork {
		let (weights, biases) = NeuralNetwork::generate_layers(&default, layerconfig);
		let network = NeuralNetwork {
			weightmats: weights,
			biases: biases,
			layerconfig: layerconfig.clone(),
			learnrate: learnrate,
			acts: &act
		};
		return network;
	}
}
impl FeedForwardRaw for NeuralNetworkInternal {}
impl<> FeedForwardInternal<NeuralNetworkInternal> for NeuralNetwork {
	fn forward_internal(&self, input: &[f32]) -> NeuralNetworkInternal {
		let mut network: NeuralNetworkInternal = NeuralNetworkInternal { layers: vec![] };
		let layer = &self.weightmats[0];
		let inputs = input.len();
		let outputs = self.layerconfig[1];
		let mut out: Vec<f32> = vec![0f32;outputs];
		for outidx in 0..outputs {
			let mut sum: f32 = self.biases[0][outidx];
			for inidx in 0..inputs {
				sum += layer[inidx + outidx * inputs] * input[inidx];
			}
			out[outidx] = self.acts.activate(sum);
		}
		network.layers.push(input.to_vec());
		network.layers.push(out);
		for layeridx in 1..self.layerconfig.len()-1 {
			// in + out * ins
			let layer = &self.weightmats[layeridx];
			let inputs = self.layerconfig[layeridx];
			let outputs = self.layerconfig[layeridx + 1];
			let mut out: Vec<f32> = vec![0f32;outputs];
			for outidx in 0..outputs {
				let mut sum: f32 = self.biases[layeridx][outidx];
				for inidx in 0..inputs {
					sum += layer[inidx + outidx * inputs] * network.layers[layeridx][inidx];
				}
				out[outidx] = self.acts.activate(sum);
			}
			network.layers.push(out);
		}
		network
	}
}
impl<> FeedForward<NeuralNetworkInternal> for NeuralNetwork {
	fn forward(&self, input: &[f32]) -> Vec<f32> {
		let mut last = input.to_vec().clone();
		for layeridx in 0..self.layerconfig.len()-1 {
			// in + out * ins
			let layer = &self.weightmats[layeridx];
			let inputs = self.layerconfig[layeridx];
			let outputs = self.layerconfig[layeridx + 1];
			let mut out: Vec<f32> = vec![0f32;outputs];
			for outidx in 0..outputs {
				let mut sum: f32 = self.biases[layeridx][outidx];
				for inidx in 0..inputs {
					sum += layer[inidx + outidx * inputs] * last[inidx];
				}
				out[outidx] = self.acts.activate(sum);
			}
			last = out;
		}
		return last;
	}
	fn backward(&mut self, input: &[f32], expected: &[f32]) -> f32{
		let internal = self.forward_internal(input);
		let weightmatcount = self.weightmats.len();
		// println!("weightmatcount={weightmatcount}");
		let mut deltas: Vec<Vec<f32>> = vec![];

		for weightmatidx in 0..weightmatcount {
			// println!("\tweightmatidx={weightmatidx}");
			deltas.push(vec![0.0f32;self.layerconfig[weightmatidx + 1]]);
		}
		// #1
		let weightmatlast = weightmatcount - 1;
		let output_activations = &internal.layers[weightmatcount];
		for actidx in 0..output_activations.len() {
			let error = output_activations[actidx] - expected[actidx];
			let act_derivative = self.acts.derivate(output_activations[actidx]);
			deltas[weightmatlast][actidx] = error * act_derivative;
		}
		// #2
		for layeridx in (0..weightmatcount - 1).rev() {
			let weights = &self.weightmats[layeridx + 1];
			let inputs = self.layerconfig[layeridx + 1]; 
			let outputs = self.layerconfig[layeridx + 2];
			let current_acts = &internal.layers[layeridx + 1];
			for inidx in 0..inputs {
				let mut sum = 0.0f32;
				for outidx in 0..outputs {
					sum += weights[inidx + outidx * inputs] * deltas[layeridx+1][outidx];
				}
				deltas[layeridx][inidx] = sum * self.acts.derivate(current_acts[inidx]);
			}
		}
		// #3
		for weightmatidx in 0..weightmatcount {
			let weights = &mut self.weightmats[weightmatidx];
			let biases = &mut self.biases[weightmatidx];
			let inputs = self.layerconfig[weightmatidx];
			let outputs = self.layerconfig[weightmatidx + 1];
			let last_values = &internal.layers[weightmatidx];
			let current_deltas = &deltas[weightmatidx];

			for outidx in 0..outputs {
				for inidx in 0..inputs {
					let gradient = current_deltas[outidx] * last_values[inidx];
					weights[inidx + outidx * inputs] -= self.learnrate * gradient;
				}
				biases[outidx] -= self.learnrate * current_deltas[outidx];
			}
		}
		let mut loss = 0.0f32;
		let output = &internal.layers.last().unwrap();
		// println!("output: {0}", output.len());
		for k in 0..output.len() {
			// println!("k: {k}");
			let d = output[k] - expected[k];
			loss += 0.5 * d * d;
		}
		loss
	}
}