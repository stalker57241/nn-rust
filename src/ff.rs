
use std::fmt::Debug;

use crate::neuro::LossType;
pub trait FeedForward<Internal, DataType = f32> {
    fn forward(&self, input: &[DataType]) -> Vec<DataType>;
    fn forward_internal(&self, input: &[DataType]) -> Internal;
}
pub trait LossCost<DataType = f32> {
    fn loss_cost(&self, losstype: LossType, expected: &[f32]) -> DataType;
}
pub trait BackPropogation<Internal, Gradients, DataType = f32>: FeedForward<Internal, DataType> {

    fn backpropogation(&self, input: &[DataType], expected: &[DataType]) -> (Vec<DataType>, Gradients);
    fn update(&mut self, gradients: &mut Gradients);
}
pub trait Activation: Debug + Send + Sync {
    fn activate(&self, layer: &[f32]) -> Vec<f32>;
    fn derivate(&self, layer: &[f32], output: &mut [f32]);
    fn vector_derivate(&self, _acts: &[f32], _expected: &[f32], _out: &mut [f32]) -> bool { return false; }
    fn get_loss_type(&self) -> LossType { LossType::Square }
    fn get_name(&self) -> String;
}