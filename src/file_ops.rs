extern crate peroxide;
extern crate reqwest;

use peroxide::fuga::*;
use std::fs::File;
use std::io;
use std::string::String;

pub async fn get_file_from_url(url: String, file_name: String) {
    let resp = reqwest::get(url).await.expect("request failed!");
    let mut out = File::create(format!("{}", file_name)).expect("failed to create file!");
    io::copy(&mut resp.text().await.unwrap().as_bytes(), &mut out).expect("failed to copy content!");
}

pub fn to_dataframe(file_path: String) -> Result<DataFrame, Box<dyn Error>> {
    let mut df = DataFrame::read_csv(&file_path.to_string(), ',')?;
    df.as_types(vec![I32, I32, Str, Str, Str, Str, Str, Str, Str, I32, I32, Str]);
    Ok(df)
}