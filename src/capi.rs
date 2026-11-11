use std::{ffi::CString, io::{Write, stderr}, os::raw::c_float, ptr::null_mut};

use crate::{
    ff::{
        Activation,
        BackPropogation,
        FeedForward
    },
    helper::{
        LEAKY_RELU,
        LINEAR,
        RELU,
        SIGMOID,
        SOFTMAX
    },
    io::{
        IONeural,
        NeuralNetworkPair
    },
    neuro::{
        LayerConfig,
        NeuralNetwork,
        NeuralNetworkGradients
    }
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum ActivationKind {
    Sigmoid = 0,
    Relu = 1,
    LeakyRelu = 2,
    Softmax = 3,
    Linear = 4,
}
impl From<usize> for ActivationKind {
    fn from(value: usize) -> Self {
        return match value {
            0 => ActivationKind::Sigmoid,
            1 => ActivationKind::Relu,
            2 => ActivationKind::LeakyRelu,
            3 => ActivationKind::Softmax,
            _ => ActivationKind::Linear
        }
    }
}
fn make_acts(cacts: Vec<usize>) -> Vec<Box<dyn Activation>> {
    let mut acts: Vec<Box<dyn Activation>> = vec![];
    for i in 0..cacts.len() {
        acts.push(match ActivationKind::from(cacts[i]) {
            ActivationKind::Linear => Box::from(LINEAR),
            ActivationKind::Sigmoid => Box::from(SIGMOID),
            ActivationKind::Relu => Box::from(RELU),
            ActivationKind::LeakyRelu => Box::from(LEAKY_RELU),
            ActivationKind::Softmax => Box::from(SOFTMAX)
        });
    }
    return  acts;
}
fn make_cacts(acts: Vec<Box<dyn Activation>>) -> Vec<usize> {
    let mut cacts: Vec<usize> = vec![];
    for i in 0..acts.len() {
        cacts.push(match acts[i].get_name().as_str() {
            "sigmoid" => 0,
            "relu" => 1,
            "leaky_relu" => 2,
            "softmax" => 3,
            _ => 4
        });
    }
    return cacts;
}
#[unsafe(no_mangle)]
pub extern "C" fn nnCreate(layers: *const usize, layer_count: usize, learnrate: c_float) -> *mut NeuralNetwork {
    if layers.is_null() || layer_count < 2 {
        return std::ptr::null_mut();
    }
    let config = unsafe { std::slice::from_raw_parts(layers, layer_count) }.to_vec();
    
    let net: NeuralNetwork = NeuralNetwork::new(&config, learnrate);
    Box::into_raw(Box::new(net))
}
#[unsafe(no_mangle)]
pub extern "C" fn nnForward(net: *mut NeuralNetwork, cacts: *const usize, inputs: *const c_float, output: *mut c_float) -> i32 {
    if net.is_null() || inputs.is_null() || output.is_null() {
        return 1;
    }
    let net = unsafe { &mut *net };
    let inputs = unsafe {std::slice::from_raw_parts(inputs, net.get_layer_config()[0]) };
    let cacts = unsafe { std::slice::from_raw_parts(cacts, net.get_layer_config().len() - 1) }.to_vec();
    let acts = &make_acts( cacts );
    let out = net.forward(acts, inputs);
    unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), output, out.len()); }
    return 0;
}
#[unsafe(no_mangle)]
pub extern "C" fn nnBackpropogate(
    net: *const NeuralNetwork,
    cacts: *const usize,
    input: *const c_float,
    expected: *const c_float
) -> *mut NeuralNetworkGradients {
    if net.is_null() || input.is_null() || expected.is_null() {
        return null_mut();
    }
    let net = unsafe { &*net };
    let input = unsafe {std::slice::from_raw_parts(input, net.get_layer_config()[0]) };
    let expected = unsafe {std::slice::from_raw_parts(expected, *net.get_layer_config().last().unwrap() ) };
    let cacts = unsafe { std::slice::from_raw_parts(cacts, net.get_layer_config().len() - 1) }.to_vec();
    let acts = &make_acts( cacts );
    let gradients = net.backpropogation(acts, input, expected).1;
    return Box::into_raw(Box::new(gradients));
}#[unsafe(no_mangle)]
pub extern "C" fn nnUpdate(net: *mut NeuralNetwork, grads: *mut NeuralNetworkGradients) {
    if net.is_null() || grads.is_null() { return; }
    let net = unsafe { &mut *net };
    let grads = unsafe { &mut *grads };
    net.update(grads);
}
#[unsafe(no_mangle)]
pub extern "C" fn nnDestroy(net: *mut NeuralNetwork) {
    if net.is_null() { return; }
    unsafe {
       drop(Box::from_raw(net));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nnStore(net: *mut NeuralNetwork, cacts: *const usize, filename: *mut i8) -> i32 {
    if net.is_null() || filename.is_null() { return false as i32; }
    let network = unsafe { &mut *net };
    let filename = unsafe {CString::from_raw(filename)}
        .to_string_lossy()
        .to_string();
    let cacts = unsafe { std::slice::from_raw_parts(cacts, network.get_layer_config().len() - 1) }.to_vec();
    let pair: NeuralNetworkPair = (network.clone(), make_acts(cacts));
   return match pair.store(filename) {
        Ok(()) => true as i32,
        Err(_) => false as i32
    };
}
#[unsafe(no_mangle)]
pub extern "C" fn nnLoad(filename: *mut i8, cacts: *mut usize) -> *mut NeuralNetwork {
    if filename.is_null() { return null_mut(); }
    let filename = unsafe { CString::from_raw(filename)}
        .to_string_lossy()
        .to_string();
    match NeuralNetworkPair::load(filename) {
        Ok((network, acts)) => {
            let _cacts = make_cacts(acts);
            unsafe {
                std::slice::from_raw_parts_mut(
                    cacts, _cacts.len()
                )
            }   .iter_mut()
                .enumerate()
                .for_each(
                    |(i, j)| *j = _cacts[i]
                );
            return Box::into_raw(Box::new(network));
        },
        Err(msg) => {
            stderr().write(&format!("Error: {msg}\n").into_bytes()).unwrap();
            return null_mut()
        }
    };
}