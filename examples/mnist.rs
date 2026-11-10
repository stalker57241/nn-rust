use nn320::mnist_minimal_required::*;

const LEARNRATE: f32 = 0.5;
const EPOCHS: usize = 7;
const BATCH_SIZE: usize = 100;

const TRAIN_IMAGES_PATH: &'static str = "mnist/train-images-idx3-ubyte";
const TRAIN_LABELS_PATH: &'static str = "mnist/train-labels-idx1-ubyte";

const T10K_IMAGES_PATH: &'static str = "mnist/t10k-images-idx3-ubyte";
const T10K_LABELS_PATH: &'static str = "mnist/t10k-labels-idx1-ubyte";

fn main() -> Result<(), std::io::Error> {
	let (
		train_image_size, 
		train_num_images, 
		train_images, 
		train_classes, 
		train_label_raw 
	) = load_images(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);
	let (
		test_image_size,
		test_num_images,
		test_images,
		test_classes,
		test_label_raw
	) = load_images(T10K_IMAGES_PATH, T10K_LABELS_PATH);
	assert_eq!(train_classes, test_classes, "TRAIN AND TEST CASES IS DIFFERENT");
	println!("Dataset loaded!");
	let mut network: NeuralNetwork = NeuralNetwork::new(
		&vec![train_image_size, 128, train_classes],
		LEARNRATE,
		vec![Box::from(SIGMOID), Box::from(SOFTMAX)]
	);
	println!("Network created!");
	for epoch in 0..EPOCHS {
		let mut total_loss = 0.0f32;
        print!(".");
        if epoch % 500 == 499 {
            println!();
        }
		for n in 0..train_num_images {
			let input: Vec<f32> = train_images[n * train_image_size..(n+1) * train_image_size].to_vec();
			let expected = one_hot(train_label_raw[n], train_classes);
			total_loss += network.backpropogation(&input, &expected).loss_cost(
				network.get_loss(),
				&expected
			);
			if network.gradients.count == BATCH_SIZE {
				network.update();
			}
		}
		println!("Epoch: {epoch}/{EPOCHS}");
		println!("loss={0}", total_loss / train_num_images as f32);
		if epoch % 5 == 0 {
			let mut correct = 0;
			let mut total_loss = 0.0f32;
			for n in 0..test_num_images {
				let input = test_images[n * test_image_size..(n + 1) * test_image_size].to_vec();
				let out = network.forward(&input);
				let expected = one_hot(test_label_raw[n], train_classes);
				for k in 0..train_classes {
					total_loss -= expected[k] * (out[k] + 1e-7).ln();
				}
				let predicted = out
					.iter()
					.enumerate()
					.max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
					.unwrap()
					.0;
				if predicted == test_label_raw[n] as usize {
					correct += 1;
				}
			}
			println!("T10K loss={0}", total_loss / test_num_images as f32);
			println!("T10K accuracy: {:.2}% ({1}/{2})", 100.0 * correct as f32 / test_num_images as f32, correct, test_num_images);
		}
	}

	let mut correct = 0;
	let mut total_loss = 0.0f32;
	for n in 0..test_num_images {
		let input = test_images[n * test_image_size..(n + 1) * test_image_size].to_vec();
		let out = network.forward(&input);
		let expected = one_hot(test_label_raw[n], train_classes);
		for k in 0..train_classes {
			// let d = out[k] - expected[k];
			// total_loss += 0.5 * d * d;
			total_loss -= expected[k] * (out[k] + 1e-7).ln();
		}
		let predicted = out
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
			.unwrap()
			.0;
		if predicted == test_label_raw[n] as usize {
			correct += 1;
		}
	}
	println!("T10K loss={0}", total_loss / test_num_images as f32);
	println!("T10K accuracy: {:.2}%", 100.0 * correct as f32 / test_num_images as f32);
	Ok(())
}
