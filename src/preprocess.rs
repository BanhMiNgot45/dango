extern crate ndarray;
extern crate polars;
extern crate vtext;

use ndarray::Array2;
use polars::prelude::*;
use std::collections::HashMap;
use std::string::String;
use vtext::tokenize::{Tokenizer, UnicodeWordTokenizer};

pub fn clean(df: DataFrame) -> DataFrame {
    let data = df.drop("")
        .unwrap()
        .drop("questionLink")
        .unwrap()
        .drop("topic")
        .unwrap()
        .drop("therapistInfo")
        .unwrap()
        .drop("therapistURL")
        .unwrap()
        .drop("split")
        .unwrap();
    let mut vec: Vec<Series> = Vec::new();
    let question_id_max = data.max().column("questionID").unwrap().sum().unwrap();
    let mut i = 0;
    loop {
        let mask_one = df.column("questionID").unwrap().eq(0);
        let frame_one = data.filter(&mask_one).unwrap();
        let upvotes_max: i32 = frame_one.max().column("upvotes").unwrap().sum().unwrap();
        let mask_two = frame_one.column("upvotes").unwrap().eq(upvotes_max);
        let frame_two = frame_one.filter(&mask_two).unwrap().sort("views", true).unwrap();
        let series = frame_two.select_at_idx(0).unwrap();
        vec.push(series.to_owned());
        i += 1;
        if i > question_id_max {
            break;
        }
    }
    DataFrame::new(vec).unwrap()
}

pub struct Text {
    pub tokens: Vec<String>
}

pub fn tokenize(df: DataFrame) -> (Vec<Text>, Vec<Text>) {
    let question = df.columns(["questionTitle", "questionText"]).unwrap();
    let answer = df.column("answerText").unwrap();
    let mut question_vec: Vec<Vec<String>> = Vec::new();
    for series in question {
        let mut vec: Vec<Vec<String>> = series.utf8().unwrap().into_iter().map(|s| to_tokens(s.unwrap().to_owned().to_lowercase())).collect();
        question_vec.append(&mut vec);
    }
    let answer_vec: Vec<Vec<String>> = answer.utf8().unwrap().into_iter().map(|s| to_tokens(s.unwrap().to_owned().to_lowercase())).collect();
    let mut questions: Vec<Text> = Vec::new();
    for vec in question_vec {
        questions.push(Text {tokens: vec});
    }
    let mut answers = Vec::new();
    for vec in answer_vec {
        answers.push(Text {tokens: vec});
    }
    (questions, answers) 
}

fn to_tokens(s: String) -> Vec<String> {
    let tokenizer = UnicodeWordTokenizer::default();
    let string = &s.to_string();
    let t: Vec<&str> = tokenizer.tokenize(string).collect();
    let mut tokens: Vec<String> = Vec::new();
    tokens.push("#BEG".to_owned());
    for string in t {
        tokens.push(string.to_owned());
    }
    tokens.push("#END".to_owned());
    tokens
}

pub fn generate_training_data(tokens: Vec<String>, word_to_id: HashMap<String, i32>, window: i32) -> (Array2<f32>, Array2<f32>) {
    let mut x: Vec<f32> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    let n = tokens.len();
    let mut i = 0;
    loop {
        let mut vec: Vec<i32> = (i32::max(0, i - window)..i).collect();
        let mut dummy: Vec<i32> = (i..i32::min(i, i + window + 1)).collect();
        vec.append(&mut dummy);
        for num in vec {
            if  i == num {
                continue;
            }
            let temp = one_hot_encode(word_to_id[tokens.get(i as usize).unwrap()], word_to_id.len() as i32);
            let dummy = one_hot_encode(word_to_id[tokens.get(num as usize).unwrap()],  word_to_id.len() as i32);
            for t in temp {
                x.push(t);
            }
            for d in dummy {
                y.push(d);
            }
        }
        i += 1;
        if i == n as i32 {
            break;
        }
    }
    (Array2::from_shape_vec((tokens.len(), word_to_id.len()), x).unwrap(), Array2::from_shape_vec((tokens.len(), word_to_id.len()), y).unwrap())
}

fn one_hot_encode(id: i32, vocab_size: i32) -> Vec<f32> {
    let mut vec: Vec<f32> = Vec::new();
    let mut i = 0;
    loop {
        if i == id {
            vec.push(1.0);
        } else {
            vec.push(0.0);
        }
        if i == vocab_size {
            break;
        }
    }
    vec
}