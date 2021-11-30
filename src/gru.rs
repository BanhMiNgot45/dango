extern crate ndarray;

use crate::math;
use ndarray::{Array2, Array3, Axis, s};
use std::collections::HashMap;

#[derive(Clone)]
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

    pub fn backward(mut self, x_out_grad: Array2<f32>, h_out_grad: Array2<f32>, mut params_dict: HashMap<String, HashMap<String, Array2<f32>>>) -> (Array2<f32>, Array2<f32>) {
        let mut temp = params_dict["B_v"]["deriv"].clone() + x_out_grad.sum_axis(Axis(0));
        params_dict.get("B_v").unwrap().to_owned().insert("deriv".to_owned(), temp);
        temp = params_dict["W_v"]["deriv"].clone() + self.h_out.t().dot(&x_out_grad);
        params_dict.get("W_v").unwrap().to_owned().insert("deriv".to_owned(), temp);
        let mut dh_out = x_out_grad.dot(&params_dict["W_v"]["value"].t());
        dh_out = dh_out + h_out_grad.clone();
        let du = self.h_in.clone() * h_out_grad.clone() - self.h_bar * h_out_grad.clone();
        let dh_bar = (Array2::<f32>::ones((self.u.nrows(), self.u.ncols())) - self.u) * h_out_grad;
        let dh_bar_int = dh_bar * math::dtanh(self.h_bar_int);
        temp = params_dict["B_h"]["deriv"].clone() + dh_bar_int.sum_axis(Axis(0));
        params_dict.get("B_h").unwrap().to_owned().insert("deriv".to_owned(), temp);
        temp = params_dict["W_xh"]["deriv"].clone() + self.x_in.t().dot(&dh_bar_int);
        params_dict.get("W_xh").unwrap().to_owned().insert("deriv".to_owned(), temp);
        let mut dx_in = dh_bar_int.dot(&params_dict.get("W_xh").unwrap().get("value").unwrap().t());
        temp = params_dict["W_hh"]["deriv"].clone() + self.h_reset.t().dot(&dh_bar_int);
        params_dict.get("W_hh").unwrap().to_owned().insert("deriv".to_owned(), temp);
        let dh_reset = dh_bar_int.dot(&params_dict["W_hh"]["value"].t());
        let dr = dh_reset.clone() * self.h_in.clone();
        let mut dh_in = dh_reset * self.r;
        let du_int = math::dsigmoid(self.u_int) * du;
        temp = params_dict["B_u"]["deriv"].clone() + du_int.sum_axis(Axis(0));
        params_dict.get("B_u").unwrap().to_owned().insert("deriv".to_owned(), temp);
        dx_in = dx_in + du_int.dot(&params_dict["W_xu"]["value"].t());
        temp = params_dict["W_xu"]["deriv"].clone() + self.x_in.t().dot(&du_int);
        params_dict.get("W_xu").unwrap().to_owned().insert("deriv".to_owned(), temp);
        dh_in = dh_in + du_int.dot(&params_dict["W_hu"]["value"].t());
        temp = self.h_in.t().dot(&du_int);
        params_dict.get("W_hu").unwrap().to_owned().insert("deriv".to_owned(), temp);
        let dr_int = math::dsigmoid(self.r_int) * dr;
        temp = params_dict["B_r"]["deriv"].clone() + dr_int.sum_axis(Axis(0));
        params_dict.get("B_r").unwrap().to_owned().insert("deriv".to_owned(), temp);
        dx_in = dx_in + dr_int.dot(&params_dict["W_xr"]["value"].t());
        temp = self.x_in.t().dot(&dr_int);
        params_dict.get("W_xr").unwrap().to_owned().insert("deriv".to_owned(), temp);
        dh_in = dh_in + dr_int.dot(&params_dict["W_hr"]["value"].t());
        temp = self.h_in.t().dot(&dr_int);
        params_dict.get("W_hr").unwrap().to_owned().insert("deriv".to_owned(), temp);
        (dx_in, dh_in)
    }
}

#[derive(Clone)]
pub struct GRULayer {
    batch_size: usize,
    vocab_size: usize,
    hidden_size: usize,
    output_size: usize,
    weight_scale: f32,
    start_h: Vec<f32>,
    first: bool,
    params_dict: HashMap<String, HashMap<String, Array2<f32>>>,
    cells: Vec<GRUCell>
}

impl GRULayer {
    pub fn new(batch_size: usize, hidden_size: usize, output_size: usize, weight_scale: f32) -> GRULayer {
        GRULayer {
            batch_size: batch_size,
            vocab_size: 0,
            hidden_size: hidden_size,
            output_size: output_size,
            weight_scale: weight_scale,
            start_h: Vec::new(),
            first: true,
            params_dict: HashMap::new(),
            cells: Vec::new()
        }
    }

    fn init_params(mut self, input: Array3<f32>) {
        self.vocab_size = input.len_of(Axis(2));
        self.params_dict.insert("W_xr".to_owned(), HashMap::new());
        self.params_dict.insert("W_hr".to_owned(), HashMap::new());
        self.params_dict.insert("B_r".to_owned(), HashMap::new());
        self.params_dict.insert("W_xu".to_owned(), HashMap::new());
        self.params_dict.insert("W_hu".to_owned(), HashMap::new());
        self.params_dict.insert("B_u".to_owned(), HashMap::new());
        self.params_dict.insert("W_xh".to_owned(), HashMap::new());
        self.params_dict.insert("W_hh".to_owned(), HashMap::new());
        self.params_dict.insert("B_h".to_owned(), HashMap::new());
        self.params_dict.insert("W_v".to_owned(), HashMap::new());
        self.params_dict.insert("B_v".to_owned(), HashMap::new());
        self.params_dict.get("W_xr").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (self.vocab_size, self.hidden_size)));
        self.params_dict.get("W_hr").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (self.hidden_size, self.hidden_size)));
        self.params_dict.get("B_r").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (1, self.hidden_size)));
        self.params_dict.get("W_xu").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (self.vocab_size, self.hidden_size)));
        self.params_dict.get("W_hu").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (self.hidden_size, self.hidden_size)));
        self.params_dict.get("B_u").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (1, self.hidden_size)));
        self.params_dict.get("W_xh").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (self.vocab_size, self.hidden_size)));
        self.params_dict.get("W_hh").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (self.hidden_size, self.hidden_size)));
        self.params_dict.get("B_h").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (1, self.hidden_size)));
        self.params_dict.get("W_v").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (self.hidden_size, self.output_size)));
        self.params_dict.get("B_v").unwrap().to_owned().insert("value".to_owned(), math::rand_array(0.0, self.weight_scale, (1, self.output_size)));
        for key in self.params_dict.keys() {
            self.params_dict.get(key).unwrap().to_owned().insert("deriv".to_owned(), Array2::zeros((self.params_dict[key]["value"].nrows(), self.params_dict[key]["value"].ncols())));
        }
        let mut i = 0;
        loop {
            self.cells.push(GRUCell::new(self.batch_size, self.hidden_size, self.vocab_size));
            i += 1;
            if i == input.len_of(Axis(1)) {
                break;
            }
        }
    }

    fn clear_gradients(self) {
        for key in self.params_dict.keys() {
            self.params_dict.get(key).unwrap().to_owned().insert("deriv".to_owned(), Array2::zeros((self.params_dict[key]["deriv"].nrows(), self.params_dict[key]["deriv"].ncols())));
        }
    }

    pub fn forward(mut self, x_seq_in: Array3<f32>) -> Array3<f32> {
        if self.first {
            self.clone().init_params(x_seq_in.clone());
            let mut a = 0;
            loop {
                self.start_h.clone().push(0.0);
                a += 1;
                if a == self.hidden_size {
                    break;
                }
            }
            self.first = false;
        }
        let mut dummy: Vec<f32> = Vec::new();
        let mut b = 0;
        loop {
            for num in self.start_h.clone() {
                dummy.push(num);
            }
            b += 1;
            if b == self.batch_size {
                break;
            }
        }
        let mut h_in = Array2::from_shape_vec((self.batch_size, self.hidden_size), dummy).unwrap();
        let sequence_length = x_seq_in.len_of(Axis(1));
        let mut x_seq_out: Vec<f32> = Vec::new();
        let mut i = 0;
        let mut y_out: Array2<f32>;
        loop {
            let x_in = x_seq_in.slice(s![.., i, ..]);
            let blah = self.cells[i].clone().forward(x_in.to_owned(), h_in.clone(), self.params_dict.clone());
            y_out = blah.0;
            h_in = blah.1;
            i += 1;
            if i == sequence_length {
                break;
            }
            for num in y_out {
                x_seq_out.push(num);
            }
        }
        self.start_h = h_in.mean_axis(Axis(0)).unwrap().to_vec();
        Array3::from_shape_vec((self.batch_size, sequence_length, self.output_size), x_seq_out).unwrap()
    }
}