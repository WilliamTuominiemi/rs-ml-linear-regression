use rand::prelude::*;

pub struct LinearRegression {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
}

impl LinearRegression {
    pub fn fit(features: Vec<Vec<f64>>, labels: Vec<f64>) -> LinearRegression {
        let (weights, bias) = Self::initialize_weights(features[0].len());

        Self {
            features,
            labels,
            weights,
            bias,
        }
    }

    pub fn train(&mut self, epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            println!("Epoch {}", epoch);

            let predictions = self.compute_predictions(&self.features, &self.weights, &self.bias);

            let loss = self.mean_squared_error(&self.labels, &predictions);

            let (weight_gradients, bias_gradient) =
                self.compute_gradients(&self.features, &self.labels, &predictions);

            for j in 0..self.weights.len() {
                self.weights[j] = self.weights[j] - learning_rate * weight_gradients[j];
            }

            self.bias = self.bias - learning_rate * bias_gradient;

            println!("Loss: {}", loss);
        }
    }

    pub fn predict(&mut self, features: Vec<Vec<f64>>) -> Vec<f64> {
        self.compute_predictions(&features, &self.weights, &self.bias)
    }

    fn mean_squared_error(&self, actual_outputs: &Vec<f64>, predicted_outputs: &Vec<f64>) -> f64 {
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

    fn initialize_weights(n: usize) -> (Vec<f64>, f64) {
        let mut rng = rand::rng();

        let mut random_weights: Vec<f64> = vec![];

        for _i in 0..n {
            let random_weight = rng.random_range(-0.001..0.001);
            random_weights.push(random_weight);
        }

        let initial_bias = 0.0;

        (random_weights, initial_bias)
    }

    fn compute_predictions(
        &self,
        features: &Vec<Vec<f64>>,
        weights: &Vec<f64>,
        bias: &f64,
    ) -> Vec<f64> {
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

    fn compute_gradients(
        &self,
        features: &Vec<Vec<f64>>,
        actual_outputs: &Vec<f64>,
        predicted_outputs: &Vec<f64>,
    ) -> (Vec<f64>, f64) {
        let n = features.len(); // number of samples
        let m = features[0].len(); // number of features

        if actual_outputs.len() != predicted_outputs.len() || features.len() != actual_outputs.len()
        {
            panic!(
                "Number of samples in features, actual_outputs, and predicted_outputs must match"
            );
        }

        let mut weight_gradients: Vec<f64> = vec![0.0; m];
        let mut bias_gradient: f64 = 0.0;

        for i in 0..n {
            let error = actual_outputs[i] - predicted_outputs[i];
            bias_gradient += error;

            for j in 0..m {
                weight_gradients[j] += features[i][j] * error;
            }
        }

        for j in 0..m {
            weight_gradients[j] *= -2.0 / n as f64;
        }
        bias_gradient *= -2.0 / n as f64;

        (weight_gradients, bias_gradient)
    }
}
