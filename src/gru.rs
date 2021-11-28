extern crate ndarray;

use crate::math;
use ndarray::{Array2};
use std::collections::HashMap;

pub struct GRUCell {
    x_in: Array2<f32>,
    h_in: Array2<f32>,
    x_out: Array2<f32>,
    h_out: Array2<f32>,
    x_r: Array2<f32>,
    h_r: Array2<f32>,
    x_u: Array2<f32>,
    h_u: Array2<f32>,
    r_int: Array2<f32>,
    r: Array2<f32>,
    u_int: f32,
    u: f32,
    h_reset: Array2<f32>,
    x_h: f32,
    h_h: f32,
    h_bar_int: f32,
    h_bar: f32
}

impl GRUCell {
    pub fn new(batch_size: usize, hidden_size: usize, vocab_size: usize) -> GRUCell {
        GRUCell {
            x_in: Array2::zeros((batch_size, vocab_size)),
            h_in: Array2::zeros((batch_size, hidden_size)),
            x_out: Array2::zeros((batch_size, vocab_size)),
            h_out: Array2::zeros((batch_size, hidden_size)),
            x_r: Array2::zeros((batch_size, hidden_size)),
            h_r: Array2::zeros((batch_size, hidden_size)),
            x_u: Array2::zeros((batch_size, hidden_size)),
            h_u: Array2::zeros((batch_size, hidden_size)),
            r_int: Array2::zeros((batch_size, hidden_size)),
            r: Array2::zeros((batch_size, hidden_size)),
            u_int: 0.0,
            u: 0.0,
            h_reset: Array2::zeros((batch_size, hidden_size)),
            x_h: 0.0,
            h_h: 0.0,
            h_bar_int: 0.0,
            h_bar: 0.0
        }
    }

    pub fn forward(mut self, x_in: Array2<f32>, h_in: Array2<f32>, params_dict: HashMap<String, HashMap<String, Array2<f32>>>) -> (Array2<f32>, Array2<f32>) {
        self.x_in = x_in;
        self.h_in = h_in;
        self.x_r = x_in.dot(params_dict.get("W_xr").unwrap().get("value").unwrap());
        self.h_r = h_in.dot(params_dict.get("W_hr").unwrap().get("value").unwrap());
        self.x_u = x_in.dot(params_dict.get("W_xu").unwrap().get("value").unwrap());
        self.h_u = h_in.dot(params_dict.get("W_hu").unwrap().get("value").unwrap());
        self.r_int = self.x_r + self.h_r + params_dict.get("B_r").unwrap().get("value").unwrap();
        self.r = math::sigmoid(self.r_int);
    }
}