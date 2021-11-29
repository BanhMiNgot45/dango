extern crate ndarray;

use ndarray::Array2;

pub fn sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|a| 1.0 / (1.0 + f32::powf(std::f32::consts::E, -a)))
}

pub fn tanh(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|a| (f32::powf(std::f32::consts::E, a) - f32::powf(std::f32::consts::E, -a)) / (f32::powf(std::f32::consts::E, a) + f32::powf(std::f32::consts::E, -a)))
}