extern crate ndarray;

use ndarray::Array2;

pub fn sigmoid(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|a| 1.0 / (1.0 + f32::powf(std::f32::consts::E, -a)))
}

pub fn tanh(arr: Array2<f32>) -> Array2<f32> {
    arr.mapv(|a| (f32::powf(std::f32::consts::E, a) - f32::powf(std::f32::consts::E, -a)) / (f32::powf(std::f32::consts::E, a) + f32::powf(std::f32::consts::E, -a)))
}

pub fn dsigmoid(arr: Array2<f32>) -> Array2<f32> {
    let temp = sigmoid(arr);
    let dummy = Array2::<f32>::ones((temp.clone().nrows(), temp.clone().ncols())) - temp.clone();
    temp * dummy
}

pub fn dtanh(arr: Array2<f32>) -> Array2<f32> {
    let temp = tanh(arr);
    let dummy = temp.mapv(|a| a.powi(2));
    Array2::<f32>::ones((dummy.nrows(), dummy.ncols())) - dummy
}