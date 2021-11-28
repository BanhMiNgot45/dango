mod file_ops;
mod one_hot_encode;
mod preprocess;
mod utils;

#[tokio::main]
async fn main() {
    let file_name = "counsel_chat.csv";
    file_ops::get_file_from_url("https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv".to_owned(), file_name.to_owned()).await;
    let df = file_ops::to_dataframe(format!("{}",file_name)).unwrap();
    let cleaned_df = preprocess::clean(df);
    let (questions, answers) = preprocess::tokenize(cleaned_df);
    let (s_to_i, i_to_s) = one_hot_encode::one_hot_encode_maps((questions, answers));
}
