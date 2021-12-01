extern crate ndarray;

use crate::math;
use ndarray::Array3;

#[derive(Clone)]
pub struct SoftmaxCrossEntropy {
    eps: f32,
    single_class: bool,
    prediction: Array3<f32>,
    target: Array3<f32>,
    output: f32,
    input_grad: Array3<f32>,
    softmax_preds: Array3<f32>
}

impl SoftmaxCrossEntropy {
    pub fn new(batch_size: usize, sequence_size: usize, vocab_size: usize) -> SoftmaxCrossEntropy {
        SoftmaxCrossEntropy {
            eps: 1e-9,
            single_class: false,
            prediction: Array3::zeros((batch_size, sequence_size, vocab_size)),
            target: Array3::zeros((batch_size, sequence_size, vocab_size)),
            output: 0.0,
            input_grad: Array3::zeros((batch_size, sequence_size, vocab_size)),
            softmax_preds: Array3::zeros((batch_size, sequence_size, vocab_size))
        }
    }

    pub fn forward(mut self, prediction: Array3<f32>, target: Array3<f32>) -> f32 {
        self.prediction = prediction;
        self.target = target;
        self.output = self.clone()._output();
        return self.output;
    }

    pub fn backward(mut self) -> Array3<f32> {
        self.input_grad = self.clone()._input_grad();
        return self.input_grad;
    }

    fn _output(self) -> f32 {
        let mut out: Vec<f32> = Vec::new();
        for row in self.prediction.outer_iter() {
            let r = math::softmax(row.to_owned());
            for num in r {
                out.push(num);
            }
        }
        let arr = Array3::from_shape_vec(self.softmax_preds.raw_dim(), out);
        let softmax_cross_entropy_loss = Array3::from_elem(self.target.raw_dim(), -1.0) * self.target.clone() * self.softmax_preds.mapv(|a| a.ln()) - (Array3::from_elem(self.target.raw_dim(), 1.0) - self.target) * self.softmax_preds.mapv(|a| (1.0 - a).ln());
        softmax_cross_entropy_loss.sum()
    }

    fn _input_grad(self) -> Array3<f32> {
        self.softmax_preds - self.target
    }
}