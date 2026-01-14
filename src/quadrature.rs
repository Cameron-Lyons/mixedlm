use pyo3::prelude::*;

pub fn gauss_hermite_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    compute_gauss_hermite(n)
}

fn compute_gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    let pi = std::f64::consts::PI;

    let m = n.div_ceil(2);

    for i in 0..m {
        let mut z = if i == 0 {
            (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-1.0 / 6.0)
        } else if i == 1 {
            let z0 = nodes[0];
            z0 - 1.14 * (n as f64).powf(0.426) / z0
        } else if i == m - 1 && n % 2 == 1 {
            0.0
        } else {
            let z_prev = nodes[i - 1];
            let z_prev2 = nodes[i - 2];
            2.0 * z_prev - z_prev2
        };

        let mut p1;
        let mut p2 = 0.0;
        for _ in 0..100 {
            let mut h0 = pi.powf(-0.25);
            let mut h1 = 0.0;

            for j in 1..=n {
                let h2 = h1;
                h1 = h0;
                h0 = z * (2.0 / j as f64).sqrt() * h1 - ((j - 1) as f64 / j as f64).sqrt() * h2;
            }

            p1 = h0;
            p2 = h1;
            let pp = (2.0 * n as f64).sqrt() * p2;

            let z_old = z;
            z -= p1 / pp;

            if (z - z_old).abs() < 1e-15 {
                break;
            }
        }

        nodes[i] = z;
        nodes[n - 1 - i] = -z;

        let w = 1.0 / (n as f64 * p2 * p2);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    let mut paired: Vec<(f64, f64)> = nodes.iter().copied().zip(weights.iter().copied()).collect();
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let nodes: Vec<f64> = paired.iter().map(|(x, _)| *x).collect();
    let weights: Vec<f64> = paired.iter().map(|(_, w)| *w).collect();

    (nodes, weights)
}

#[pyfunction]
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    gauss_hermite_nodes_weights(n)
}

#[pyfunction]
pub fn adaptive_gauss_hermite_1d(
    nodes: Vec<f64>,
    weights: Vec<f64>,
    mode: f64,
    scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let sqrt2 = std::f64::consts::SQRT_2;

    let adapted_nodes: Vec<f64> = nodes.iter().map(|&x| mode + sqrt2 * scale * x).collect();

    let adapted_weights: Vec<f64> = weights
        .iter()
        .zip(nodes.iter())
        .map(|(&w, &x)| w * (x * x).exp())
        .collect();

    (adapted_nodes, adapted_weights)
}
