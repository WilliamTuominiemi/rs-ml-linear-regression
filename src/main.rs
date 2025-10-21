use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Deserialize)]
struct Datapoint {
    tv: f64,        // feature
    radio: f64,     // feature
    newspaper: f64, // feature
    sales: f64,     // label
}

fn main() -> Result<(), Box<dyn Error>> {
    let dataponts = read_data()?;

    println!("{:?}", dataponts);

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
