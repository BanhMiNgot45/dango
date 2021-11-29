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
    u_int: Array2<f32>,
    u: Array2<f32>,
    h_reset: Array2<f32>,
    x_h: Array2<f32>,
    h_h: Array2<f32>,
    h_bar_int: Array2<f32>,
    h_bar: Array2<f32>
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
            u_int: Array2::zeros((batch_size, hidden_size)),
            u: Array2::zeros((batch_size, hidden_size)),
            h_reset: Array2::zeros((batch_size, hidden_size)),
            x_h: Array2::zeros((batch_size, hidden_size)),
            h_h: Array2::zeros((batch_size, hidden_size)),
            h_bar_int: Array2::zeros((batch_size, hidden_size)),
            h_bar: Array2::zeros((batch_size, hidden_size))
        }
    }

    pub fn forward(mut self, x_in: Array2<f32>, h_in: Array2<f32>, params_dict: HashMap<String, HashMap<String, Array2<f32>>>) -> (Array2<f32>, Array2<f32>) {
        self.x_in = x_in;
        self.h_in = h_in;
        self.x_r = self.x_in.dot(params_dict.get("W_xr").unwrap().get("value").unwrap());
        self.h_r = self.h_in.dot(params_dict.get("W_hr").unwrap().get("value").unwrap());
        self.x_u = self.x_in.dot(params_dict.get("W_xu").unwrap().get("value").unwrap());
        self.h_u = self.h_in.dot(params_dict.get("W_hu").unwrap().get("value").unwrap());
        self.r_int = self.x_r.clone() + self.h_r.clone() + params_dict.get("B_r").unwrap().get("value").unwrap();
        self.r = math::sigmoid(self.r_int);
        self.u_int = self.x_r + self.h_r + params_dict.get("B_u").unwrap().get("value").unwrap();
        self.u = math::sigmoid(self.u_int);
        self.h_reset = self.r * self.h_in.clone();
        self.x_h = self.x_in.dot(params_dict.get("W_xh").unwrap().get("value").unwrap());
        self.h_h = self.h_reset.dot(params_dict.get("W_hh").unwrap().get("value").unwrap());
        self.h_bar_int = self.x_h + self.h_h + params_dict.get("B_h").unwrap().get("value").unwrap();
        self.h_bar = math::tanh(self.h_bar_int);
        self.h_out = self.u.clone() * self.h_in + (Array2::<f32>::ones((self.u.nrows(), self.u.ncols())) - self.u) * self.h_bar;
        self.x_out = self.h_out.dot(params_dict.get("W_v").unwrap().get("value").unwrap()) + params_dict.get("B_v").unwrap().get("value").unwrap();
        (self.h_out, self.x_out)
    }
}