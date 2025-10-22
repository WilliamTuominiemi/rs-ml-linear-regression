use std::collections::btree_map::Range;

use rand::{prelude::*, rng};

pub struct LinearRegression {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
    rng: ThreadRng,
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

        let rng = rand::rng();

        Self {
            features: biased_features,
            labels,
            rng,
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

    fn initialize_weights(&mut self, n: usize) -> (Vec<f64>, f64) {
        let mut random_weights: Vec<f64> = vec![];

        for i in 0..n {
            let random_weight = self.rng.random_range(0.0..1.0);
            random_weights.push(random_weight);
        }

        let random_bias = self.rng.random_range(0.0..1.0);
        (random_weights, random_bias)
    }

    fn compute_predictions(features: Vec<Vec<f64>>, weights: Vec<f64>, bias: f64) -> Vec<f64> {
        let n = features.len(); // number of samples
        let m = features[0].len(); // number of feature

        if m != weights.len() {
            panic!("Features and weights have different length");
        }

        let mut predictions: Vec<f64> = vec![];

        for j in 0..n {
            let mut prediction = 0.0;
            for i in 0..m {
                prediction += weights[i] * features[j][i]
            }
            prediction += bias;
            predictions.push(prediction);
        }

        predictions
    }
}
