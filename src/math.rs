extern crate ndarray;
extern crate rand;
extern crate rand_distr;

use ndarray::{Array, Array2};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

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

pub fn rand_array(loc: f32, scale: f32, size: (usize, usize)) -> Array2<f32> {
    let mut temp: Vec<f32> = Vec::new();
    let normal = Normal::new(loc, scale).unwrap();
    let mut i = 0;
    loop {
        let mut j = 0;
        loop {
            temp.push(normal.sample(&mut rand::thread_rng()));
            j += 1;
            if j == size.1 {
                break;
            }
        }
        i += 1;
        if i == size.0 {
            break;
        }
    }
    Array::from_shape_vec(size, temp).unwrap()
}

pub fn softmax(arr: Array2<f32>) -> Array2<f32> {
    let mut vec: Vec<f32> = Vec::new();
    for row in arr.outer_iter() {
        let mut accum = 0.0;
        for num in row {
            accum += f32::powf(std::f32::consts::E, num.to_owned());
        }
        let new_row = row.mapv(|a| a / accum);
        for n in new_row {
            vec.push(n);
        }
    }
    Array2::from_shape_vec((arr.nrows(), arr.ncols()), vec).unwrap()
}