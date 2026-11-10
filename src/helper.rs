use crate::{ff::Activation, neuro::LossType};

pub mod io {
    use std::fs;

    pub fn read_idx_f32(p: &str) -> (Vec<usize>, Vec<f32>){
        let bytes = fs::read(p).expect("Read failed!");
        let dtype = bytes[2];
        let ndims = bytes[3] as usize;
        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let offset = 4 + i * 4;
            let d = u32::from_be_bytes([
                bytes[offset + 0],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3]
            ]) as usize;
            dims.push(d);
        }
        let header_len = 4 + ndims * 4;
        let data_types = &bytes[header_len..];
        let data: Vec<f32> = match dtype {
            0x0D => data_types
                .chunks_exact(4)
                .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            0x08 => data_types.iter().map(|&b| b as f32).collect(),
            0x09 => data_types.iter().map(|&b| (b as i8) as f32).collect(),
            _ => panic!("Unsupported type: {dtype:#x}")
        };
        return (dims, data);
    }
    pub fn read_idx(p: &str) -> (Vec<usize>, Vec<u8>){
        let bytes = fs::read(p).expect("Read failed!");
        let dtype = bytes[2];
        let ndims = bytes[3] as usize;
        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let offset = 4 + i * 4;
            let d = u32::from_be_bytes([
                bytes[offset + 0],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3]
            ]) as usize;
            dims.push(d);
        }
        let header_len = 4 + ndims * 4;
        let data = bytes[header_len..].to_vec();
        
        if dtype != 0x08 {
            panic!("This loader only for unsigned byte, got {dtype:#x}");
        }
        return (dims, data);
    }

    pub fn load_images(images_path: &str, labels_path: &str) -> (usize, usize, Vec<f32>, usize, Vec<u8>) {
        let (dims, raw) = read_idx_f32(images_path);
        let num_images = dims[0];
        let rows = dims[1];
        let cols = dims[2];
        let image_size = rows * cols; // 784

        let images: Vec<f32> = raw.iter().map(|&b| b as f32 / 255.0).collect();

        let (label_dims, label_raw) = read_idx(labels_path);
        let num_labels = label_dims[0];
        let max_label = *label_raw.iter().max().unwrap();
        let classes = (max_label as usize) + 1;
        assert_eq!(num_images, num_labels);
        (image_size, num_images, images, classes, label_raw)
    }
}

pub fn softmax(z: &[f32]) -> Vec<f32> {
	let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	let exps: Vec<f32> = z.iter().map(|&x| (x - max_z).exp()).collect();
	let sum: f32 = exps.iter().sum();
	exps.iter().map(|&e| e / sum).collect()
}

#[derive(Debug, Clone, Copy)]
pub struct Activatable(fn (f32) -> f32, fn (f32) -> f32, LossType, &'static str);
impl Activation for Activatable {
	fn activate(&self, layer: &[f32]) -> Vec<f32> {
		return layer.iter().map(|x| self.0(*x)).collect();
	}
	fn derivate(&self, layer: &[f32], output: &mut [f32]) { 
        if layer.len() == output.len() {
            layer.iter().enumerate().for_each(|(idx, x)| output[idx] = self.1(*x));
        }
	}
    fn get_loss_type(&self) -> LossType {
        return self.2;
    }
    fn get_name(&self) -> String {
        return self.3.to_string();
    }
}
#[derive(Debug, Clone, Copy)]
pub struct SoftMax;
impl Activation for SoftMax {
	fn activate(&self, layer: &[f32]) -> Vec<f32> {
		return softmax(layer);
	}
	fn derivate(&self, _layer: &[f32], _output: &mut [f32]) {
		panic!("softmax derivate не используется, см. vector_derivate")
	}
	fn get_loss_type(&self) -> LossType { LossType::CrossEntropy }
	fn vector_derivate(&self, acts: &[f32], expected: &[f32], out: &mut [f32]) -> bool {
        // if acts.len() != expected.len() || acts.len() != out.len() {
        //     return false;
        // }
		for actidx in 0..acts.len() {
			out[actidx] = acts[actidx] - expected[actidx];
		}
		return true;
	}
    fn get_name(&self) -> String {
        return "softmax".to_string()
    }
}
pub const SIGMOID: Activatable = Activatable(
	|x| 1.0 / (1.0 + (-x).exp()),
	|x| x * (1.0 - x),
	LossType::Square,
    "sigmoid"
);
pub const RELU: Activatable = Activatable(
	|x| if x > 0.0f32 {x} else {0.0f32},
	|x| if x > 0.0f32 {1.0f32} else {0.0f32},
	LossType::Square,
    "relu"
);
pub const LEAKY_RELU: Activatable = Activatable(
	|x| if x > 0.0f32 {x} else {0.01f32 * x},
	|x| if x > 0.0f32 {1.0f32} else {0.01f32},
	LossType::Square,
    "leaky_relu"
);
pub const SOFTMAX: SoftMax = SoftMax;
pub const LINEAR: Activatable = Activatable(
    |x| x,
    |_| 1f32,
    LossType::Square,
    "linear"
);
pub fn one_hot(label: u8, classes: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; classes];
    v[label as usize] = 1.0;
    v
}