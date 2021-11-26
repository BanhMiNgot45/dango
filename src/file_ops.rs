extern crate reqwest;
extern crate polars;

use polars::prelude::*;
use std::fs::File;
use std::io;
use std::string::String;

pub async fn get_file_from_url(url: String, file_name: String) {
    let resp = reqwest::get(url).await.expect("request failed!");
    let mut out = File::create(format!("{}", file_name)).expect("failed to create file!");
    io::copy(&mut resp.text().await.unwrap().as_bytes(), &mut out).expect("failed to copy content!");
}

pub fn to_dataframe(file_path: String) -> Result<DataFrame> {
    CsvReader::from_path(file_path)?.infer_schema(None).has_header(true).finish()
}