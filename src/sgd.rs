extern crate ndarray;

use crate::gru::GRUModel;
use ndarray::Array2;

#[derive(Clone)]
pub struct SGD {
    lr: f32,
    first: bool,
    model: GRUModel
}

impl SGD {
    pub fn new(lr: f32, model: GRUModel) -> SGD {
        SGD {
            lr: lr,
            first: true,
            model: model
        }
    }

    pub fn step(self) {
        for layer in self.clone().model.layers {
            for key in layer.params_dict.keys() {
                self.clone()._update(&mut layer.params_dict.get(key).unwrap().get("value").unwrap().to_owned(), layer.params_dict.get(key).unwrap().get("deriv").unwrap().to_owned());
            }
        }
    }

    fn _update(self, param: &mut Array2<f32>, grad: Array2<f32>) {
        let update = self.lr * grad;
        *param = param.to_owned() - update;
    }
}