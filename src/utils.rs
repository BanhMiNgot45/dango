use std::collections::HashMap;
use std::cmp::{max, min};
use std::string::String;

fn n_grams(vec: Vec<String>, n: i32) -> Vec<Vec<String>> {
    let mut i = 0;
    let mut ngram: Vec<Vec<String>> = Vec::new();
    loop {
        ngram.push(vec[i..(i + n as usize)].to_vec());
        i += 1;
    }
}

fn count_n_grams(vec: Vec<String>, n: i32) -> HashMap<Vec<String>, i32> {
    let mut map: HashMap<Vec<String>, i32> = HashMap::new();
    let mut i = 0;
    let ngram = n_grams(vec, n);
    loop {
        let vec = &ngram[i];
        if map.contains_key(&vec.to_owned()) {
            continue;
        }
        let mut counter = 0;
        let mut j = 0;
        loop {
            if vec[i] == vec[j] {
                counter += 1;
            }
            j += 1;
            if j == vec.len() {

                break;
            }
        }
        map.insert(vec.to_owned(), counter);
        i += 1;
        if i == vec.len() {
            break;
        }
    }
    map
}

fn count_clip_ngram(translation: Vec<String>, list_of_references: Vec<Vec<String>>, n: i32) -> HashMap<Vec<String>, i32> {
    let mut map: HashMap<Vec<String>, i32> = HashMap::new();
    let ct_translation = count_n_grams(translation, n);
    for references in list_of_references {
        let ct_reference = count_n_grams(references, n);
        for ct in ct_reference.keys() {
            if map.contains_key(ct) {
                map.insert(ct.to_owned(), max(ct_reference.get(ct).unwrap().to_owned(), map.get(ct).unwrap().to_owned()));
            } else {
                map.insert(ct.to_owned(), ct_reference.get(ct).unwrap().to_owned());
            }
        }
    }
    let mut mapping: HashMap<Vec<String>, i32> = HashMap::new();
    for ct in ct_translation.keys() {
        let mut a = -1;
        if ct_translation.contains_key(ct) {
            a = ct_translation.get(ct).unwrap().to_owned();
        } else {
            a = 0;
        }
        let mut b = -1;
        if map.contains_key(ct) {
            b = map.get(ct).unwrap().to_owned();
        } else {
            b = 0;
        }
        mapping.insert(ct.to_owned(), min(a, b));
    }
    mapping
}

fn modified_precision(translation: Vec<String>, list_of_references: Vec<Vec<String>>, n: i32) -> f32 {
    let ct_clip = count_clip_ngram(translation.clone(), list_of_references, n);
    let ct = count_n_grams(translation, n);
    let mut ct_clip_cumulative = 0;
    for c in ct_clip {
        ct_clip_cumulative += c.1;
    }
    let mut ct_cumulative = 0;
    for c in ct {
        ct_cumulative += c.1;
    }
    ct_clip_cumulative as f32 / max(ct_cumulative, 1) as f32
}

fn closest_ref_length(translation: Vec<String>, list_of_references: Vec<Vec<String>>) -> i32 {
    let translation_length = translation.len();
}