use pyo3::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

pub fn gauss_hermite_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![1.7724538509055159]),
        2 => (
            vec![-FRAC_1_SQRT_2, FRAC_1_SQRT_2],
            vec![0.8862269254527579, 0.8862269254527579],
        ),
        3 => (
            vec![-1.224_744_871_391_589, 0.0, 1.224_744_871_391_589],
            vec![0.2954089751509193, 1.1816359006036772, 0.2954089751509193],
        ),
        5 => (
            vec![
                -2.0201828704560856,
                -0.9585724646138185,
                0.0,
                0.9585724646138185,
                2.0201828704560856,
            ],
            vec![
                0.01995324205904591,
                0.3936193231522411,
                0.9453087204829419,
                0.3936193231522411,
                0.01995324205904591,
            ],
        ),
        7 => (
            vec![
                -2.6519613568352334,
                -1.6735516287674714,
                -0.8162878828589647,
                0.0,
                0.8162878828589647,
                1.6735516287674714,
                2.6519613568352334,
            ],
            vec![
                0.0009717812450995192,
                0.05451558281912703,
                0.4256072526101277,
                0.8102646175568073,
                0.4256072526101277,
                0.05451558281912703,
                0.0009717812450995192,
            ],
        ),
        9 => (
            vec![
                -3.1909932017815276,
                -2.266_580_584_531_843,
                -1.468_553_289_216_668,
                -0.7235510187528376,
                0.0,
                0.7235510187528376,
                1.468_553_289_216_668,
                2.266_580_584_531_843,
                3.1909932017815276,
            ],
            vec![
                0.00002234584400044273,
                0.002788806661191671,
                0.04991640676521788,
                0.244_097_502_894_939_1,
                0.4060142980038694,
                0.244_097_502_894_939_1,
                0.04991640676521788,
                0.002788806661191671,
                0.00002234584400044273,
            ],
        ),
        11 => (
            vec![
                -3.668_470_846_559_582,
                -2.783_290_099_781_652,
                -2.0259480158257553,
                -1.3265570844949329,
                -0.6568095668820998,
                0.0,
                0.6568095668820998,
                1.3265570844949329,
                2.0259480158257553,
                2.783_290_099_781_652,
                3.668_470_846_559_582,
            ],
            vec![
                1.4394934e-06,
                0.0003467116,
                0.011911395,
                0.11726396,
                0.36954569,
                0.46224370,
                0.36954569,
                0.11726396,
                0.011911395,
                0.0003467116,
                1.4394934e-06,
            ],
        ),
        _ => {
            let (nodes, weights) = compute_gauss_hermite(n);
            (nodes, weights)
        }
    }
}

fn compute_gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    let pi = std::f64::consts::PI;

    for i in 0..n {
        let mut z = if i == 0 {
            (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-0.16667)
        } else if i == 1 {
            nodes[0] - 1.14 * (n as f64).powf(0.426) / nodes[0]
        } else if i == n - 1 {
            -nodes[0]
        } else if i == n - 2 {
            -nodes[1]
        } else {
            nodes[i - 1] - (nodes[i - 1] - nodes[i - 2])
        };

        let mut p2_final = 0.0;
        for _ in 0..100 {
            let mut p1 = pi.powf(-0.25);
            let mut p2 = 0.0;

            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                p1 = z * (2.0 / j as f64).sqrt() * p2 - ((j - 1) as f64 / j as f64).sqrt() * p3;
            }

            p2_final = p2;
            let pp = (2.0 * n as f64).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;

            if (z - z1).abs() < 1e-14 {
                break;
            }
        }

        nodes[i] = z;
        weights[i] = 2.0 / (p2_final * p2_final * n as f64);
    }

    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let sqrt_pi = pi.sqrt();
    for i in 0..n {
        let mut p1 = std::f64::consts::PI.powf(-0.25);
        let mut p2 = 0.0;

        for j in 1..=n {
            let p3 = p2;
            p2 = p1;
            p1 = nodes[i] * (2.0 / j as f64).sqrt() * p2 - ((j - 1) as f64 / j as f64).sqrt() * p3;
        }

        weights[i] = sqrt_pi / (n as f64 * p2 * p2);
    }

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
