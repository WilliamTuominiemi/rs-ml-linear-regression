use serde::Deserialize;
use std::error::Error;
use std::vec;

mod linear_regression;
use linear_regression::LinearRegression;

#[derive(Debug, Deserialize)]
struct Datapoint {
    tv: f64,        // feature
    radio: f64,     // feature
    newspaper: f64, // feature
    sales: f64,     // label
}

fn main() -> Result<(), Box<dyn Error>> {
    let datapoints = read_data()?;

    let x: Vec<Vec<f64>> = datapoints
        .iter()
        .map(|dp| vec![dp.tv, dp.radio, dp.newspaper])
        .collect();
    let y: Vec<f64> = datapoints.iter().map(|dp| dp.sales).collect();

    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2);

    let mut linear_regression = LinearRegression::fit(x_train, y_train);

    linear_regression.train(50, 0.05);

    let predictions = linear_regression.predict(x_test);

    println!("Predicted {:?}", predictions);
    println!("True {:?}", y_test);

    Ok(())
}

fn read_data() -> Result<Vec<Datapoint>, Box<dyn Error>> {
    let mut datapoints = vec![];

    let mut rdr = csv::Reader::from_path("advertising.csv")?;
    for result in rdr.deserialize() {
        let datapoint: Datapoint = result?;
        datapoints.push(datapoint);
    }

    Ok(datapoints)
}

fn train_test_split(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    test_size: f64,
) -> (
    Vec<Vec<f64>>, // x_train
    Vec<Vec<f64>>, // x_test
    Vec<f64>,      // y_train
    Vec<f64>,      // y_test
) {
    let dataset_size = x.len();
    if dataset_size != y.len() {
        panic!("x and y must be same length");
    }

    let split_at = ((1.0 - test_size) * dataset_size as f64) as usize;

    let (x_train, x_test) = x.split_at(split_at);
    let (y_train, y_test) = y.split_at(split_at);

    (
        x_train.to_vec(),
        x_test.to_vec(),
        y_train.to_vec(),
        y_test.to_vec(),
    )
}
