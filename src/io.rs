use std::{fs::File, io::{self, Read, Write}, path::Path, sync::Arc};

use crate::{ff::Activation, helper::{LEAKY_RELU, LINEAR, RELU, SIGMOID, SOFTMAX}, neuro::{LayerConfig, NeuralNetwork}};

pub trait IONeural<T> {
    fn exists(filename: String) -> bool {
        Path::new(filename.as_str()).exists()
    }
    fn store(&self, filename: String) -> Result<(), io::Error>;
    fn load(filename: String) -> Result<T, io::Error>;
    fn load_activation(index: u8) -> Box<dyn Activation>;
}
impl IONeural<NeuralNetwork> for NeuralNetwork {
    fn load_activation(index: u8) -> Box<dyn Activation> {
        match index {
            1 => return Box::from(SIGMOID),
            2 => return Box::from(RELU),
            3 => return Box::from(LEAKY_RELU),
            4 => return Box::from(SOFTMAX),
            _ => return Box::from(LINEAR)
        }
    }
    fn store(&self, filename: String) -> Result<(), io::Error> {
        // lr
        // layers.len()
        // [layers]
        // [acts]
        // [[weightmat][biases]]
        let network = &self;
        let acts = &self.acts;
        let mut file = File::create(filename)?;
        file.write_all(&network.learnrate.to_le_bytes())?;
        file.write_all(&network.get_layer_config().len().to_le_bytes())?;
        for i in 0..network.get_layer_config().len() { // [layers]
            file.write_all(&network.get_layer_config()[i].to_le_bytes())?;
        }
        let weightmatcount = network.get_layer_config().len() - 1;
        for matidx in 0..weightmatcount { // [acts]
            file.write_all(&match acts[matidx].get_name().as_str() {
                "sigmoid" => 1u8.to_le_bytes(),
                "relu" => 2u8.to_le_bytes(),
                "leaky_relu" => 3u8.to_le_bytes(),
                "softmax" => 4u8.to_le_bytes(),
                _ => 0u8.to_le_bytes()
            })?;
        }
        for matidx in 0..weightmatcount { // [[weightmat][biases]]
            let inputs = network.get_layer_config()[matidx];
            let outputs = network.get_layer_config()[matidx + 1];
            let weightcount = inputs * outputs;
            let biascount = outputs;
            for widx in 0..weightcount { // [weightmat]
                file.write_all(&network.weightmats[matidx][widx].to_le_bytes())?;
            } 
            for bidx in 0..biascount {// [biases]
                file.write_all(&network.weightmats[matidx][bidx].to_le_bytes())?;
            }
        }
        
        Ok(())
    }
    fn load(filename: String) -> Result<NeuralNetwork, io::Error> {
        // lr
        // layers.len()
        // [layers]
        // [acts]
        // [[weightmat][biases]]
        let mut file = File::open(filename)?;
        let mut u8_buf = [0u8; std::mem::size_of::<u8>()];
        let mut f32_buf = [0u8; std::mem::size_of::<f32>()];
        let mut usize_buf = [0u8; std::mem::size_of::<usize>()];
        file.read_exact(&mut f32_buf)?;
        let lr = f32::from_le_bytes(f32_buf);
        file.read_exact(&mut usize_buf)?;
        let layercount = usize::from_le_bytes(usize_buf);
        let mut layers: Vec<usize> = vec![0;layercount];
        for layeridx in 0..layercount {
            file.read_exact(&mut usize_buf)?;
            layers[layeridx] = usize::from_le_bytes(usize_buf);
        }
        let matrix_count = layercount - 1;
        let mut acts: Vec<Box<dyn Activation>> = vec![];
        for matidx in 0..matrix_count {
            file.read_exact(&mut u8_buf)?;
            acts.insert(matidx, Self::load_activation(u8::from_le_bytes(u8_buf)));
        }
        let mut weightmats: Vec<Vec<f32>> = vec![vec![];matrix_count];
        let mut biases: Vec<Vec<f32>> = vec![vec![];matrix_count];
        let mut bias_modulations: Vec<f32> = vec![1.0f32; matrix_count];
        for matidx in 0..matrix_count {
            let nextmatidx = matidx + 1;
            let inputs = layers[matidx];
            let outputs = layers[nextmatidx];
            let weightcount = inputs * outputs;
            let biascount = outputs;
            weightmats[matidx] = vec![0f32;weightcount];
            biases[matidx] = vec![0f32;biascount];
            bias_modulations[matidx] = 1.0f32;
            for widx in 0..weightcount {
                file.read_exact(&mut f32_buf)?;
                weightmats[matidx][widx] = f32::from_le_bytes(f32_buf);
            }
            for bidx in 0..biascount {
                file.read_exact(&mut f32_buf)?;
                biases[matidx][bidx] = f32::from_le_bytes(f32_buf);
            }
            file.read_exact(&mut f32_buf)?;
            bias_modulations[matidx] = f32::from_le_bytes(f32_buf);
        }
        let network = NeuralNetwork {
            layerconfig: layers.clone(),
            weightmats: weightmats,
            biases: biases,
            bias_modulation: bias_modulations,
            learnrate: lr,
            acts: Arc::from(acts)
        };
        Ok(network)
    }
    
    fn exists(filename: String) -> bool {
        Path::new(filename.as_str()).exists()
    }
}