use crate::preprocess::Text;
use std::collections::{HashMap, HashSet};
use std::string::String;

pub fn one_hot_encode_maps(text_blobs: (Vec<Text>, Vec<Text>)) -> (HashMap<String, i32>, HashMap<i32, String>) {
    let mut set: HashSet<String> = HashSet::new();
    for text in text_blobs.0 {
        for s in text.tokens {
            set.insert(s);
        }
    }
    for text in text_blobs.1 {
        for s in text.tokens {
            set.insert(s);
        }
    }
    let mut i = 3;
    let mut s_to_i: HashMap<String, i32> = HashMap::new();
    s_to_i.insert("#BEG".to_owned(), 0);
    s_to_i.insert("#END".to_owned(), 1);
    s_to_i.insert("#UNK".to_owned(), 2);
    let mut i_to_s: HashMap<i32, String> = HashMap::new();
    i_to_s.insert(0, "#BEG".to_owned());
    i_to_s.insert(1, "#END".to_owned());
    i_to_s.insert(2, "#UNK".to_owned());
    for word in set {
        s_to_i.insert(word.clone(), i);
        i_to_s.insert(i, word);
        i += 1;
    }
    (s_to_i, i_to_s)
}

pub fn encode(vec: Vec<String>, map: HashMap<String, i32>) -> Vec<i32> {
    let mut vector: Vec<i32> = Vec::new();
    for s in vec {
        vector.push(map.get(&s).unwrap().to_owned());
    }
    vector
}

pub fn decode(vec: Vec<i32>, map: HashMap<i32, String>) -> Vec<String> {
    let mut vector: Vec<String> = Vec::new();
    for i in vec {
        vector.push(map.get(&i).unwrap().to_owned());
    }
    vector
}