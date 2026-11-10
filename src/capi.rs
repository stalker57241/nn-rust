use std::{ffi::CString, io::{Write, stderr}, os::raw::c_float, ptr::null_mut};

use crate::{ff::{Activation, BackPropogation, FeedForward}, helper::{LEAKY_RELU, LINEAR, RELU, SIGMOID, SOFTMAX}, io::IONeural, neuro::{LayerConfig, NeuralNetwork}};

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

#[unsafe(no_mangle)]
pub extern "C" fn nnCreate(layers: *const usize, cacts: *const usize, layer_count: usize, learnrate: c_float) -> *mut NeuralNetwork {
    if layers.is_null() || layer_count < 2 {
        return std::ptr::null_mut();
    }
    let config = unsafe { std::slice::from_raw_parts(layers, layer_count) }.to_vec();
    let cacts = unsafe { std::slice::from_raw_parts(cacts, layer_count - 1)}.to_vec();
    let mut acts: Vec<Box<dyn Activation>> = vec![];
    for i in 0..layer_count-1 {
        acts.push(match ActivationKind::from(cacts[i]) {
            ActivationKind::Linear => Box::from(LINEAR),
            ActivationKind::Sigmoid => Box::from(SIGMOID),
            ActivationKind::Relu => Box::from(RELU),
            ActivationKind::LeakyRelu => Box::from(LEAKY_RELU),
            ActivationKind::Softmax => Box::from(SOFTMAX)
        });
    }

    let net: NeuralNetwork = NeuralNetwork::new(&config, learnrate, acts);
    Box::into_raw(Box::new(net))
}
#[unsafe(no_mangle)]
pub extern "C" fn nnForward(net: *mut NeuralNetwork, inputs: *const c_float, output: *mut c_float) -> i32 {
    if net.is_null() || inputs.is_null() || output.is_null() {
        return 1;
    }
    let net = unsafe { &mut *net };
    let inputs = unsafe {std::slice::from_raw_parts(inputs, net.get_layer_config()[0]) };
    let out = net.forward(inputs);
    unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), output, out.len()); }
    return 0;
}
#[unsafe(no_mangle)]
pub extern "C" fn nnBackpropogate(
    net: *mut NeuralNetwork,
    input: *const c_float,
    expected: *const c_float
) -> i32 {
    if net.is_null() || input.is_null() || expected.is_null() {
        return 1;
    }
    let net = unsafe { &mut *net };
    let input = unsafe {std::slice::from_raw_parts(input, net.get_layer_config()[0]) };
    let expected = unsafe {std::slice::from_raw_parts(expected, *net.get_layer_config().last().unwrap() ) };
    net.backpropogation(input, expected);
    return 0;
}#[unsafe(no_mangle)]
pub extern "C" fn nnUpdate(net: *mut NeuralNetwork) {
    if net.is_null() { return; }
    let net = unsafe { &mut *net };
    net.update();
}

#[unsafe(no_mangle)]
pub extern "C" fn nnDestroy(net: *mut NeuralNetwork) {
    if net.is_null() { return; }
    unsafe {
       drop(Box::from_raw(net));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nnStore(net: *mut NeuralNetwork, filename: *mut i8) -> i32 {
    if net.is_null() || filename.is_null() { return false as i32; }
    let net = unsafe { &mut *net };
    let filename = unsafe {CString::from_raw(filename)}
        .to_string_lossy()
        .to_string();
    return match net.store(filename) {
        Ok(()) => true as i32,
        Err(_) => false as i32
    };
}
#[unsafe(no_mangle)]
pub extern "C" fn nnLoad(filename: *mut i8) -> *mut NeuralNetwork {
    if filename.is_null() { return null_mut(); }
    let filename = unsafe { CString::from_raw(filename)}
        .to_string_lossy()
        .to_string();
    match NeuralNetwork::load(filename) {
        Ok(network) => return Box::into_raw(Box::new(network)),
        Err(msg) => {
            stderr().write(&format!("Error: {msg}\n").into_bytes()).unwrap();
            return null_mut()
        }
    };
}