use std::result;

pub struct LinearRegression {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
}

impl LinearRegression {
    pub fn fit(features: Vec<Vec<f64>>, labels: Vec<f64>) -> LinearRegression {
        let biased_features: Vec<Vec<f64>> = features
            .iter()
            .map(|feature| {
                let mut biased_feature = feature.clone();
                biased_feature.push(1.0);
                biased_feature
            })
            .collect();

        Self {
            features: biased_features,
            labels,
        }
    }

    fn mean_squared_error(actual_outputs: Vec<f64>, predicted_outputs: Vec<f64>) -> f64 {
        let n = actual_outputs.len();
        if n != predicted_outputs.len() {
            panic!("Actual and predicted outputs need to be same length");
        }

        let mut result = 0.0;

        for i in 0..n {
            let difference = actual_outputs[i] - predicted_outputs[i];
            let squared_difference = difference.powf(2.0);
            result += squared_difference
        }

        let mean = result / n as f64;
        mean
    }
}
