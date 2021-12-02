use crate::gru;
use crate::loss;
use crate::wordvec;

pub struct EnsembleModel {
    embeddings: wordvec::WordVec,
    encoder: gru::GRUModel,
    decoder: gru::GRUModel
}

impl EnsembleModel {
    pub fn new(embedding_size: i32, vocab_size: i32, hidden_size: i32, batch_size: i32, sequence_length: i32, output_size: i32, weight_scale: f32) -> EnsembleModel {
        EnsembleModel {
            embeddings: wordvec::WordVec::new(vocab_size, embedding_size),
            encoder: gru::GRUModel::new(vec![gru::GRULayer::new(batch_size as usize, hidden_size as usize, output_size as usize, weight_scale)], sequence_length, vocab_size, hidden_size, loss::SoftmaxCrossEntropy::new(batch_size as usize, sequence_length as usize, vocab_size as usize)),
            decoder: gru::GRUModel::new(vec![gru::GRULayer::new(batch_size as usize, hidden_size as usize, output_size as usize, weight_scale)], sequence_length, vocab_size, hidden_size, loss::SoftmaxCrossEntropy::new(batch_size as usize, sequence_length as usize, vocab_size as usize))
        }
    }
}