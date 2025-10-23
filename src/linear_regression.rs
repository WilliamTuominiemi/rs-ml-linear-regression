use rand::prelude::*;

pub struct LinearRegression {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
    weights: Vec<f64>,
    bias: f64,
    means: Vec<f64>,
    standard_deviations: Vec<f64>,
}

impl LinearRegression {
    pub fn fit(features: Vec<Vec<f64>>, labels: Vec<f64>) -> LinearRegression {
        let (weights, bias) = Self::initialize_weights(features[0].len());

        let means = LinearRegression::means(&features);
        let standard_deviations = LinearRegression::standard_deviation(&features, &means);
        let normalized_features =
            LinearRegression::normalize(&features, &means, &standard_deviations);

        Self {
            features: normalized_features,
            labels,
            weights,
            bias,
            means,
            standard_deviations,
        }
    }

    pub fn train(&mut self, epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            println!("Epoch {}", epoch + 1);

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
        let n = features.len();
        let m = features[0].len();

        let mut normalized_features: Vec<Vec<f64>> = vec![];

        for i in 0..n {
            let mut normalized_feature: Vec<f64> = vec![];
            for j in 0..m {
                normalized_feature
                    .push((features[i][j] - self.means[j]) / self.standard_deviations[j]);
            }
            normalized_features.push(normalized_feature);
        }

        self.compute_predictions(&normalized_features, &self.weights, &self.bias)
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
            let random_weight = rng.random_range(-0.01..0.01);
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

    fn means(features: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut means: Vec<f64> = vec![];

        let n = features.len();
        let m = features[0].len();

        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += features[j][i];
            }
            means.push(sum / n as f64);
        }

        means
    }

    fn standard_deviation(features: &Vec<Vec<f64>>, means: &Vec<f64>) -> Vec<f64> {
        let mut standard_deviations: Vec<f64> = vec![];

        let n = features.len();
        let m = features[0].len();

        for i in 0..m {
            let mut sqaured_differences = 0.0;
            for j in 0..n {
                let difference = features[j][i] - means[i];
                sqaured_differences += difference * difference;
            }
            standard_deviations.push((sqaured_differences / n as f64).sqrt());
        }

        standard_deviations
    }

    fn normalize(
        features: &Vec<Vec<f64>>,
        means: &Vec<f64>,
        standard_deviations: &Vec<f64>,
    ) -> Vec<Vec<f64>> {
        let n = features.len();
        let m = features[0].len();

        let mut normalized_features: Vec<Vec<f64>> = vec![];

        for i in 0..n {
            let mut normalized_feature: Vec<f64> = vec![];
            for j in 0..m {
                normalized_feature.push((features[i][j] - means[j]) / standard_deviations[j]);
            }
            normalized_features.push(normalized_feature);
        }

        normalized_features
    }
}
