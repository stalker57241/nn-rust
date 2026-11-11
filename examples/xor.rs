use nn320::io::IONeural;
use nn320::ff::Activation;
use nn320::neuro::GetLoss;
use nn320::{helper::LEAKY_RELU, minimal_required::*};
use nn320::io::NeuralNetworkPair;

const TRAIN_COUNT: usize = 4;
const INS: usize = 2;
const TRAIN_SET: [([f32;INS], u8);TRAIN_COUNT] = [
    ([0f32, 0f32], 0),
    ([0f32, 1f32], 1),
    ([1f32, 0f32], 1),
    ([1f32, 1f32], 0),
];
const OUTS: usize = 2;

const EPOCHS: usize = 1;
const LEARNRATE: f32 = 0.05f32;

const BATCH_SIZE: usize = 100;
fn main() {
    let layerconfig: Vec<usize> = vec![INS, 2, OUTS];
    let name: String = format!("xor_{}_LR{LEARNRATE}.mdl", layerconfig.iter().map(|x| return x.to_string()).collect::<Vec<String>>().join("_"));
    let acts: Vec<Box<dyn Activation>> = vec![Box::from(LEAKY_RELU), Box::from(SOFTMAX)];
    let (mut network, acts) = NeuralNetworkPair::load(name.clone()).unwrap_or(
        (NeuralNetwork::new(&layerconfig, LEARNRATE), acts)
    );
    let losstype = acts.get_loss();
    let mut last_correct = 0;
    let mut decades = 0usize;
    while last_correct < TRAIN_COUNT {
        for epoch in 0..EPOCHS {
            let mut total_loss = 0.0f32;
            for n in 0..TRAIN_COUNT {
                let input: Vec<f32> = TRAIN_SET[n].0.to_vec();
                let expected = one_hot(TRAIN_SET[n].1, OUTS);
                let (out, mut gradient) = network.backpropogation(&acts, &input, &expected);
                total_loss += out.loss_cost(
                    losstype,
                    &expected
                );
                if gradient.count == BATCH_SIZE {
                    network.update(&mut gradient);
                }
            }
            if epoch % 1000 == 999 {
                println!("Epoch: {epoch}/{EPOCHS} loss={0}", total_loss / TRAIN_COUNT as f32);
            }
        }
        let mut correct: usize = 0;
        let mut total_loss = 0f32;
        for example in 0..TRAIN_SET.len() {
            // let XOR_TRAIN_SET = [];
            let input = TRAIN_SET[example].0.to_vec();
            let out = network.forward(&acts, &input);
            let expected = one_hot(TRAIN_SET[example].1, OUTS);
            for k in 0..OUTS { total_loss -= expected[k] * (out[k] + 1e-7).ln(); }
            let predicted = out.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if predicted == TRAIN_SET[example].1 as usize {
                correct += 1;
            }
        }
        // network.update();
        println!("Total:\nloss: {0}, correct: {1}/{2}", total_loss, correct, TRAIN_SET.len());
        last_correct = correct;
        decades += 1;
    }
    println!("Decades needed: {decades} (CYCLES OF EPOCHS TO COMPLETE 100% CORRECT)");
    (network, acts).store(name.clone()).unwrap();
}