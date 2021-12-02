extern crate ndarray;
extern crate rand;

use crate::math;
use ndarray::Array2;
use rand::Rng;
use std::collections::HashMap;
use std::string::String;

pub struct WordVec {
    vocab_size: i32,
    n_embeddings: i32,
    w1: Array2<f32>,
    w2: Array2<f32>,
}

impl WordVec {
    pub fn new(vocab_size: i32, n_embeddings: i32) -> WordVec {
        let mut temp: Vec<f32> = Vec::new();
        let mut dummy: Vec<f32> = Vec::new();
        let mut i = 0;
        let mut rng = rand::thread_rng();
        loop {
            temp.push(rng.gen::<f32>());
            dummy.push(rng.gen::<f32>());
            i += 1;
            if i == vocab_size * n_embeddings {
                break;
            }
        }
        WordVec {
            vocab_size: vocab_size,
            n_embeddings: n_embeddings,
            w1: Array2::from_shape_vec((vocab_size as usize, n_embeddings as usize), temp).unwrap(),
            w2: Array2::from_shape_vec((n_embeddings as usize, vocab_size as usize), dummy).unwrap()
        }
    }
}

pub fn forward(model: &mut WordVec, x: Array2<f32>) -> HashMap<String, Array2<f32>> {
    let mut cache: HashMap<String, Array2<f32>> = HashMap::new();
    cache.insert("a1".to_owned(), x.dot(&model.w1));
    cache.insert("a2".to_owned(), cache["a1"].dot(&model.w2));
    cache.insert("z".to_owned(), math::softmax(cache["a2"].to_owned()));
    cache
}

pub fn backward(model: &mut WordVec, x: Array2<f32>, y: Array2<f32>, alpha: f32) -> f32 {
    let cache = forward(model, x.clone());
    let da2 = cache.get("z").unwrap() - y.clone();
    let dw2 = cache["a1"].t().dot(&da2);
    let da1 = da2.dot(&model.w2.t());
    let dw1 = x.t().dot(&da1);
    model.w1 = model.w1.to_owned() - alpha * dw1;
    model.w2 = model.w2.to_owned() - alpha * dw2;
    (cache.get("z").unwrap().mapv(|a| a.ln()) * y).sum()
}