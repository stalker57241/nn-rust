use std::fs;

use crate::neuro::{Act, FeedForward, NeuralNetwork, Randomf32};


pub mod neuro;


static SIGMOID: Act = Act (
	|x| 1.0 / (1.0 + (-x).exp()),
	|x| x * (1.0 - x)
);
const LEARNRATE: f32 = 0.5;
const EPOCHS: usize = 350;
const DECAYED_EPOCHS: usize = 335;
fn read_idx_f32(p: &str) -> (Vec<usize>, Vec<f32>){
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
fn read_idx(p: &str) -> (Vec<usize>, Vec<u8>){
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

const TRAIN_IMAGES_PATH: &'static str = "mnist/train-images-idx3-ubyte";
const TRAIN_LABELS_PATH: &'static str = "mnist/train-labels-idx1-ubyte";

const T10K_IMAGES_PATH: &'static str = "mnist/t10k-images-idx3-ubyte";
const T10K_LABELS_PATH: &'static str = "mnist/t10k-labels-idx1-ubyte";
fn one_hot(label: u8, classes: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; classes];
    v[label as usize] = 1.0;
    v
}
fn load_images(images_path: &str, labels_path: &str) -> (usize, usize, Vec<f32>, usize, Vec<u8>) {
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

fn main() -> Result<(), std::io::Error> {
	let (
		image_size,
		num_images,
		images,
		classes,
		label_raw
	) = load_images(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);


	let mut network: NeuralNetwork = NeuralNetwork::new(
		Randomf32(-0.5, 0.5),
		&vec![image_size, 192, 96, classes],
		LEARNRATE,
		&SIGMOID
	);
	
	for epoch in 0..EPOCHS {
		let mut total_loss = 0.0f32;
		for n in 0..num_images {
			let input: Vec<f32> = images[n * image_size..(n+1) * image_size].to_vec();
			let expected = one_hot(label_raw[n], classes);

			total_loss += network.backward(&input, &expected);
		}
		if epoch >= EPOCHS - DECAYED_EPOCHS {
			network.degrade_learnrate(epoch, DECAYED_EPOCHS as f32 / 3.0, LEARNRATE, 0.001f32);
			println!("learnrate degradated: {}", network.get_learnrate());
		}
		println!("Epoch: {epoch}/{EPOCHS}: loss={0}", total_loss / num_images as f32);
	}
	let (
		image_size,
		num_images,
		images,
		_,
		label_raw
	) = load_images(T10K_IMAGES_PATH, T10K_LABELS_PATH);
	let mut correct = 0;
	let mut total_loss = 0.0f32;
	for n in 0..num_images {
		let input = images[n * image_size..(n + 1) * image_size].to_vec();
		let out = network.forward(&input);
		let expected = one_hot(label_raw[n], classes);
		for k in 0..classes {
			let d = out[k] - expected[k];
			total_loss += 0.5 * d * d;
		}
		let predicted = out
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
			.unwrap()
			.0;
		if predicted == label_raw[n] as usize {
			correct += 1;
		}
	}
	println!("T10K loss={0}", total_loss / num_images as f32);
	println!("T10K accuracy: {:.2}%", 100.0 * correct as f32 / num_images as f32);

	Ok(())
}
