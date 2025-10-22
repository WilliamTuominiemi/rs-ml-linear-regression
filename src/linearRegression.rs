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
}
