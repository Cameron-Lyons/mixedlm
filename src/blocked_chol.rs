use faer::linalg::solvers::Llt;
use faer::{Mat, Side};
use rayon::prelude::*;

use crate::linalg::LinalgError;
use crate::lmm::RandomEffectStructure;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum BlockType {
    Dense(Mat<f64>),
    Diagonal(Vec<f64>),
    BlockDiagonal {
        block_size: usize,
        blocks: Vec<Mat<f64>>,
    },
    Zero {
        rows: usize,
        cols: usize,
    },
}

#[allow(dead_code)]
impl BlockType {
    pub fn rows(&self) -> usize {
        match self {
            BlockType::Dense(m) => m.nrows(),
            BlockType::Diagonal(d) => d.len(),
            BlockType::BlockDiagonal { block_size, blocks } => block_size * blocks.len(),
            BlockType::Zero { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            BlockType::Dense(m) => m.ncols(),
            BlockType::Diagonal(d) => d.len(),
            BlockType::BlockDiagonal { block_size, blocks } => block_size * blocks.len(),
            BlockType::Zero { cols, .. } => *cols,
        }
    }

    pub fn to_dense(&self) -> Mat<f64> {
        match self {
            BlockType::Dense(m) => m.clone(),
            BlockType::Diagonal(d) => {
                let n = d.len();
                Mat::from_fn(n, n, |i, j| if i == j { d[i] } else { 0.0 })
            }
            BlockType::BlockDiagonal { block_size, blocks } => {
                let n = block_size * blocks.len();
                let mut result = Mat::zeros(n, n);
                for (k, block) in blocks.iter().enumerate() {
                    let offset = k * block_size;
                    for i in 0..*block_size {
                        for j in 0..*block_size {
                            result[(offset + i, offset + j)] = block[(i, j)];
                        }
                    }
                }
                result
            }
            BlockType::Zero { rows, cols } => Mat::zeros(*rows, *cols),
        }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            BlockType::Dense(m) => m[(i, j)],
            BlockType::Diagonal(d) => {
                if i == j {
                    d[i]
                } else {
                    0.0
                }
            }
            BlockType::BlockDiagonal { block_size, blocks } => {
                let bi = i / block_size;
                let bj = j / block_size;
                if bi == bj {
                    let li = i % block_size;
                    let lj = j % block_size;
                    blocks[bi][(li, lj)]
                } else {
                    0.0
                }
            }
            BlockType::Zero { .. } => 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockedMatrix {
    pub block_dims: Vec<usize>,
    pub blocks: Vec<Vec<BlockType>>,
}

impl BlockedMatrix {
    pub fn from_lambda_ztwz(
        ztwz: &Mat<f64>,
        lambda_blocks: &[Mat<f64>],
        structures: &[RandomEffectStructure],
        add_identity: bool,
    ) -> Self {
        let n_blocks = structures.len();
        let block_dims: Vec<usize> = structures.iter().map(|s| s.n_levels * s.n_terms).collect();

        let mut block_offsets = vec![0usize];
        for dim in &block_dims {
            block_offsets.push(block_offsets.last().unwrap() + dim);
        }

        let mut blocks: Vec<Vec<BlockType>> = Vec::with_capacity(n_blocks);

        for i in 0..n_blocks {
            let mut row_blocks: Vec<BlockType> = Vec::with_capacity(i + 1);
            let qi = structures[i].n_terms;
            let ni = structures[i].n_levels;
            let lambda_i = &lambda_blocks[i];
            let lambda_i_t = lambda_i.transpose();

            for j in 0..=i {
                let qj = structures[j].n_terms;
                let nj = structures[j].n_levels;
                let lambda_j = &lambda_blocks[j];
                let offset_i = block_offsets[i];
                let offset_j = block_offsets[j];

                if i == j {
                    let mut sub_blocks: Vec<Mat<f64>> = Vec::with_capacity(ni);

                    for level in 0..ni {
                        let li = level * qi;
                        let mut block = Mat::zeros(qi, qi);

                        for ii in 0..qi {
                            for jj in 0..qi {
                                block[(ii, jj)] = ztwz[(offset_i + li + ii, offset_i + li + jj)];
                            }
                        }

                        let transformed = lambda_i_t.as_ref() * &block * lambda_i;

                        let mut result_block = Mat::zeros(qi, qi);
                        for ii in 0..qi {
                            for jj in 0..qi {
                                result_block[(ii, jj)] = transformed[(ii, jj)];
                            }
                        }

                        if add_identity {
                            for ii in 0..qi {
                                result_block[(ii, ii)] += 1.0;
                            }
                        }

                        sub_blocks.push(result_block);
                    }

                    row_blocks.push(BlockType::BlockDiagonal {
                        block_size: qi,
                        blocks: sub_blocks,
                    });
                } else {
                    let mut dense_block = Mat::zeros(block_dims[i], block_dims[j]);

                    for level_i in 0..ni {
                        let li = level_i * qi;
                        for level_j in 0..nj {
                            let lj = level_j * qj;

                            let mut block = Mat::zeros(qi, qj);
                            for ii in 0..qi {
                                for jj in 0..qj {
                                    block[(ii, jj)] =
                                        ztwz[(offset_i + li + ii, offset_j + lj + jj)];
                                }
                            }

                            let transformed = lambda_i_t.as_ref() * &block * lambda_j;

                            for ii in 0..qi {
                                for jj in 0..qj {
                                    dense_block[(li + ii, lj + jj)] = transformed[(ii, jj)];
                                }
                            }
                        }
                    }

                    let is_zero = dense_block
                        .col_iter()
                        .all(|col| col.iter().all(|&v| v.abs() < 1e-15));

                    if is_zero {
                        row_blocks.push(BlockType::Zero {
                            rows: block_dims[i],
                            cols: block_dims[j],
                        });
                    } else {
                        row_blocks.push(BlockType::Dense(dense_block));
                    }
                }
            }

            blocks.push(row_blocks);
        }

        BlockedMatrix { block_dims, blocks }
    }

    #[allow(dead_code)]
    pub fn to_dense(&self) -> Mat<f64> {
        let total_dim: usize = self.block_dims.iter().sum();
        let mut result = Mat::zeros(total_dim, total_dim);

        let mut block_offsets = vec![0usize];
        for dim in &self.block_dims {
            block_offsets.push(block_offsets.last().unwrap() + dim);
        }

        for (i, row_blocks) in self.blocks.iter().enumerate() {
            for (j, block) in row_blocks.iter().enumerate() {
                let offset_i = block_offsets[i];
                let offset_j = block_offsets[j];
                let dense = block.to_dense();

                for ii in 0..dense.nrows() {
                    for jj in 0..dense.ncols() {
                        result[(offset_i + ii, offset_j + jj)] = dense[(ii, jj)];
                        if i != j {
                            result[(offset_j + jj, offset_i + ii)] = dense[(ii, jj)];
                        }
                    }
                }
            }
        }

        result
    }
}

#[derive(Debug)]
pub struct BlockedCholesky {
    block_dims: Vec<usize>,
    l_blocks: Vec<Vec<BlockType>>,
}

impl BlockedCholesky {
    #[allow(clippy::needless_range_loop)]
    pub fn factor(a: &BlockedMatrix) -> Result<Self, LinalgError> {
        let n_blocks = a.block_dims.len();
        let block_dims = a.block_dims.clone();
        let mut l_blocks: Vec<Vec<BlockType>> = Vec::with_capacity(n_blocks);

        for i in 0..n_blocks {
            let mut row_blocks: Vec<BlockType> = Vec::with_capacity(i + 1);

            for j in 0..=i {
                if i == j {
                    let mut aii = a.blocks[i][i].clone();

                    for k in 0..i {
                        let lik = &row_blocks[k];
                        rank_update_subtract(&mut aii, lik, lik)?;
                    }

                    let lii = chol_block(&aii)?;
                    row_blocks.push(lii);
                } else {
                    let mut aij = a.blocks[i][j].clone();

                    for k in 0..j {
                        let lik = &row_blocks[k];
                        let ljk = &l_blocks[j][k];
                        rank_update_subtract(&mut aij, lik, ljk)?;
                    }

                    let ljj = &l_blocks[j][j];
                    let lij = forward_solve_block_transpose(ljj, &aij)?;
                    row_blocks.push(lij);
                }
            }

            l_blocks.push(row_blocks);
        }

        Ok(BlockedCholesky {
            block_dims,
            l_blocks,
        })
    }

    pub fn solve(&self, b: &Mat<f64>) -> Mat<f64> {
        let y = self.forward_solve(b);
        self.backward_solve(&y)
    }

    fn forward_solve(&self, b: &Mat<f64>) -> Mat<f64> {
        let total_dim: usize = self.block_dims.iter().sum();
        let ncols = b.ncols();
        let mut y = Mat::zeros(total_dim, ncols);

        let mut block_offsets = vec![0usize];
        for dim in &self.block_dims {
            block_offsets.push(block_offsets.last().unwrap() + dim);
        }

        for (i, row_blocks) in self.l_blocks.iter().enumerate() {
            let offset_i = block_offsets[i];
            let dim_i = self.block_dims[i];

            let mut rhs = Mat::zeros(dim_i, ncols);
            for ii in 0..dim_i {
                for c in 0..ncols {
                    rhs[(ii, c)] = b[(offset_i + ii, c)];
                }
            }

            for (j, lij) in row_blocks.iter().enumerate().take(i) {
                let offset_j = block_offsets[j];
                let dim_j = self.block_dims[j];

                let mut yj = Mat::zeros(dim_j, ncols);
                for jj in 0..dim_j {
                    for c in 0..ncols {
                        yj[(jj, c)] = y[(offset_j + jj, c)];
                    }
                }

                let contrib = block_matvec(lij, &yj);
                for ii in 0..dim_i {
                    for c in 0..ncols {
                        rhs[(ii, c)] -= contrib[(ii, c)];
                    }
                }
            }

            let lii = &row_blocks[i];
            let yi = solve_lower_block(lii, &rhs);

            for ii in 0..dim_i {
                for c in 0..ncols {
                    y[(offset_i + ii, c)] = yi[(ii, c)];
                }
            }
        }

        y
    }

    #[allow(clippy::needless_range_loop)]
    fn backward_solve(&self, y: &Mat<f64>) -> Mat<f64> {
        let total_dim: usize = self.block_dims.iter().sum();
        let ncols = y.ncols();
        let mut x = Mat::zeros(total_dim, ncols);

        let mut block_offsets = vec![0usize];
        for dim in &self.block_dims {
            block_offsets.push(block_offsets.last().unwrap() + dim);
        }

        let n_blocks = self.l_blocks.len();
        for i in (0..n_blocks).rev() {
            let offset_i = block_offsets[i];
            let dim_i = self.block_dims[i];

            let mut rhs = Mat::zeros(dim_i, ncols);
            for ii in 0..dim_i {
                for c in 0..ncols {
                    rhs[(ii, c)] = y[(offset_i + ii, c)];
                }
            }

            for j in (i + 1)..n_blocks {
                let offset_j = block_offsets[j];
                let dim_j = self.block_dims[j];
                let lji = &self.l_blocks[j][i];

                let mut xj = Mat::zeros(dim_j, ncols);
                for jj in 0..dim_j {
                    for c in 0..ncols {
                        xj[(jj, c)] = x[(offset_j + jj, c)];
                    }
                }

                let contrib = block_matvec_transpose(lji, &xj);
                for ii in 0..dim_i {
                    for c in 0..ncols {
                        rhs[(ii, c)] -= contrib[(ii, c)];
                    }
                }
            }

            let lii = &self.l_blocks[i][i];
            let xi = solve_lower_transpose_block(lii, &rhs);

            for ii in 0..dim_i {
                for c in 0..ncols {
                    x[(offset_i + ii, c)] = xi[(ii, c)];
                }
            }
        }

        x
    }

    pub fn logdet(&self) -> f64 {
        let mut logdet = 0.0;
        for (i, row_blocks) in self.l_blocks.iter().enumerate() {
            let lii = &row_blocks[i];
            logdet += block_logdet(lii);
        }
        2.0 * logdet
    }
}

fn chol_block(block: &BlockType) -> Result<BlockType, LinalgError> {
    match block {
        BlockType::Dense(m) => {
            let chol =
                Llt::new(m.as_ref(), Side::Lower).map_err(|_| LinalgError::NotPositiveDefinite)?;
            Ok(BlockType::Dense(chol.L().to_owned()))
        }
        BlockType::Diagonal(d) => {
            let l: Result<Vec<f64>, LinalgError> = d
                .iter()
                .map(|&x| {
                    if x <= 0.0 {
                        Err(LinalgError::NotPositiveDefinite)
                    } else {
                        Ok(x.sqrt())
                    }
                })
                .collect();
            Ok(BlockType::Diagonal(l?))
        }
        BlockType::BlockDiagonal { block_size, blocks } => {
            let results: Result<Vec<Mat<f64>>, LinalgError> = blocks
                .par_iter()
                .map(|b| {
                    let chol = Llt::new(b.as_ref(), Side::Lower)
                        .map_err(|_| LinalgError::NotPositiveDefinite)?;
                    Ok(chol.L().to_owned())
                })
                .collect();

            Ok(BlockType::BlockDiagonal {
                block_size: *block_size,
                blocks: results?,
            })
        }
        BlockType::Zero { rows, cols } => Ok(BlockType::Zero {
            rows: *rows,
            cols: *cols,
        }),
    }
}

fn rank_update_subtract_inplace(target: &mut BlockType, l: &BlockType, r: &BlockType) {
    if matches!(l, BlockType::Zero { .. }) || matches!(r, BlockType::Zero { .. }) {
        return;
    }
    if matches!(target, BlockType::Zero { .. }) {
        return;
    }

    match (&*target, l, r) {
        (BlockType::Dense(_), BlockType::Dense(l_mat), BlockType::Dense(r_mat)) => {
            let update = l_mat * r_mat.transpose();
            if let BlockType::Dense(t) = target {
                for i in 0..t.nrows() {
                    for j in 0..t.ncols() {
                        t[(i, j)] -= update[(i, j)];
                    }
                }
            }
        }
        (
            BlockType::BlockDiagonal { .. },
            BlockType::BlockDiagonal {
                blocks: l_blocks, ..
            },
            BlockType::BlockDiagonal {
                blocks: r_blocks, ..
            },
        ) => {
            if let BlockType::BlockDiagonal {
                block_size: bs,
                blocks: t_blocks,
            } = target
            {
                let bs = *bs;
                for (k, t_block) in t_blocks.iter_mut().enumerate() {
                    let update = &l_blocks[k] * r_blocks[k].transpose();
                    for i in 0..bs {
                        for j in 0..bs {
                            t_block[(i, j)] -= update[(i, j)];
                        }
                    }
                }
            }
        }
        _ => {
            let l_dense = l.to_dense();
            let r_dense = r.to_dense();
            let update = &l_dense * r_dense.transpose();

            if let BlockType::Dense(t) = target {
                for i in 0..t.nrows() {
                    for j in 0..t.ncols() {
                        t[(i, j)] -= update[(i, j)];
                    }
                }
            }
        }
    }
}

fn rank_update_subtract(
    target: &mut BlockType,
    l: &BlockType,
    r: &BlockType,
) -> Result<(), LinalgError> {
    if matches!(l, BlockType::Zero { .. }) || matches!(r, BlockType::Zero { .. }) {
        return Ok(());
    }
    if matches!(target, BlockType::Zero { .. }) {
        return Ok(());
    }

    let needs_conversion = matches!(
        (&*target, l, r),
        (BlockType::BlockDiagonal { .. }, BlockType::Dense(_), _)
            | (BlockType::BlockDiagonal { .. }, _, BlockType::Dense(_))
    );

    if needs_conversion {
        let mut dense_target = target.to_dense();
        let l_dense = l.to_dense();
        let r_dense = r.to_dense();
        let update = &l_dense * r_dense.transpose();

        for i in 0..dense_target.nrows() {
            for j in 0..dense_target.ncols() {
                dense_target[(i, j)] -= update[(i, j)];
            }
        }

        *target = BlockType::Dense(dense_target);
    } else {
        rank_update_subtract_inplace(target, l, r);
    }

    Ok(())
}

fn solve_lower(l: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let n = l.nrows();
    let ncols = b.ncols();
    let mut x = b.clone();

    for c in 0..ncols {
        for i in 0..n {
            for k in 0..i {
                x[(i, c)] -= l[(i, k)] * x[(k, c)];
            }
            x[(i, c)] /= l[(i, i)];
        }
    }
    x
}

#[allow(dead_code)]
fn solve_lower_transpose(l: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let n = l.nrows();
    let ncols = b.ncols();
    let mut x = b.clone();

    for c in 0..ncols {
        for i in (0..n).rev() {
            for k in (i + 1)..n {
                x[(i, c)] -= l[(k, i)] * x[(k, c)];
            }
            x[(i, c)] /= l[(i, i)];
        }
    }
    x
}

fn forward_solve_block_transpose(l: &BlockType, b: &BlockType) -> Result<BlockType, LinalgError> {
    match (l, b) {
        (
            BlockType::BlockDiagonal {
                block_size: bs,
                blocks: l_blocks,
            },
            BlockType::Dense(b_mat),
        ) => {
            let nrows = b_mat.nrows();
            let ncols = b_mat.ncols();
            let mut result = Mat::zeros(nrows, ncols);

            let bs = *bs;
            for (k, l_block) in l_blocks.iter().enumerate() {
                let col_offset = k * bs;

                for row in 0..nrows {
                    let mut col_vec = Mat::zeros(bs, 1);
                    for j in 0..bs {
                        col_vec[(j, 0)] = b_mat[(row, col_offset + j)];
                    }

                    let solved = solve_lower(l_block, &col_vec);

                    for j in 0..bs {
                        result[(row, col_offset + j)] = solved[(j, 0)];
                    }
                }
            }

            Ok(BlockType::Dense(result))
        }
        (BlockType::Dense(l_mat), BlockType::Dense(b_mat)) => {
            let nrows = b_mat.nrows();
            let ncols = b_mat.ncols();
            let mut result = Mat::zeros(nrows, ncols);

            for row in 0..nrows {
                let mut col_vec = Mat::zeros(ncols, 1);
                for j in 0..ncols {
                    col_vec[(j, 0)] = b_mat[(row, j)];
                }

                let solved = solve_lower(l_mat, &col_vec);

                for j in 0..ncols {
                    result[(row, j)] = solved[(j, 0)];
                }
            }

            Ok(BlockType::Dense(result))
        }
        (_, BlockType::Zero { rows, cols }) => Ok(BlockType::Zero {
            rows: *rows,
            cols: *cols,
        }),
        (BlockType::Diagonal(d), BlockType::Dense(b_mat)) => {
            let nrows = b_mat.nrows();
            let ncols = b_mat.ncols();
            let mut result = Mat::zeros(nrows, ncols);

            for i in 0..nrows {
                for j in 0..ncols {
                    result[(i, j)] = b_mat[(i, j)] / d[j];
                }
            }

            Ok(BlockType::Dense(result))
        }
        _ => {
            let l_dense = l.to_dense();
            let b_dense = b.to_dense();
            let nrows = b_dense.nrows();
            let ncols = b_dense.ncols();
            let mut result = Mat::zeros(nrows, ncols);

            for row in 0..nrows {
                let mut col_vec = Mat::zeros(ncols, 1);
                for j in 0..ncols {
                    col_vec[(j, 0)] = b_dense[(row, j)];
                }

                let solved = solve_lower(&l_dense, &col_vec);

                for j in 0..ncols {
                    result[(row, j)] = solved[(j, 0)];
                }
            }

            Ok(BlockType::Dense(result))
        }
    }
}

fn block_matvec(block: &BlockType, v: &Mat<f64>) -> Mat<f64> {
    match block {
        BlockType::Dense(m) => m * v,
        BlockType::Diagonal(d) => Mat::from_fn(v.nrows(), v.ncols(), |i, j| d[i] * v[(i, j)]),
        BlockType::BlockDiagonal { block_size, blocks } => {
            let mut result = Mat::zeros(v.nrows(), v.ncols());
            for (k, b) in blocks.iter().enumerate() {
                let offset = k * block_size;
                for i in 0..*block_size {
                    for j in 0..v.ncols() {
                        let mut sum = 0.0;
                        for l in 0..*block_size {
                            sum += b[(i, l)] * v[(offset + l, j)];
                        }
                        result[(offset + i, j)] = sum;
                    }
                }
            }
            result
        }
        BlockType::Zero { rows, .. } => Mat::zeros(*rows, v.ncols()),
    }
}

fn block_matvec_transpose(block: &BlockType, v: &Mat<f64>) -> Mat<f64> {
    match block {
        BlockType::Dense(m) => m.transpose() * v,
        BlockType::Diagonal(d) => Mat::from_fn(v.nrows(), v.ncols(), |i, j| d[i] * v[(i, j)]),
        BlockType::BlockDiagonal { block_size, blocks } => {
            let mut result = Mat::zeros(v.nrows(), v.ncols());
            for (k, b) in blocks.iter().enumerate() {
                let offset = k * block_size;
                for i in 0..*block_size {
                    for j in 0..v.ncols() {
                        let mut sum = 0.0;
                        for l in 0..*block_size {
                            sum += b[(l, i)] * v[(offset + l, j)];
                        }
                        result[(offset + i, j)] = sum;
                    }
                }
            }
            result
        }
        BlockType::Zero { cols, .. } => Mat::zeros(*cols, v.ncols()),
    }
}

fn solve_lower_block(l: &BlockType, b: &Mat<f64>) -> Mat<f64> {
    match l {
        BlockType::Dense(m) => {
            let mut y = b.clone();
            for i in 0..m.nrows() {
                for c in 0..b.ncols() {
                    for k in 0..i {
                        y[(i, c)] -= m[(i, k)] * y[(k, c)];
                    }
                    y[(i, c)] /= m[(i, i)];
                }
            }
            y
        }
        BlockType::Diagonal(d) => Mat::from_fn(b.nrows(), b.ncols(), |i, j| b[(i, j)] / d[i]),
        BlockType::BlockDiagonal { block_size, blocks } => {
            let mut result = Mat::zeros(b.nrows(), b.ncols());
            for (k, block) in blocks.iter().enumerate() {
                let offset = k * block_size;

                for c in 0..b.ncols() {
                    for i in 0..*block_size {
                        let mut val = b[(offset + i, c)];
                        for j in 0..i {
                            val -= block[(i, j)] * result[(offset + j, c)];
                        }
                        result[(offset + i, c)] = val / block[(i, i)];
                    }
                }
            }
            result
        }
        BlockType::Zero { rows, .. } => Mat::zeros(*rows, b.ncols()),
    }
}

fn solve_lower_transpose_block(l: &BlockType, b: &Mat<f64>) -> Mat<f64> {
    match l {
        BlockType::Dense(m) => {
            let mut x = b.clone();
            let n = m.nrows();
            for i in (0..n).rev() {
                for c in 0..b.ncols() {
                    for k in (i + 1)..n {
                        x[(i, c)] -= m[(k, i)] * x[(k, c)];
                    }
                    x[(i, c)] /= m[(i, i)];
                }
            }
            x
        }
        BlockType::Diagonal(d) => Mat::from_fn(b.nrows(), b.ncols(), |i, j| b[(i, j)] / d[i]),
        BlockType::BlockDiagonal { block_size, blocks } => {
            let mut result = Mat::zeros(b.nrows(), b.ncols());
            for (k, block) in blocks.iter().enumerate() {
                let offset = k * block_size;

                for c in 0..b.ncols() {
                    for i in (0..*block_size).rev() {
                        let mut val = b[(offset + i, c)];
                        for j in (i + 1)..*block_size {
                            val -= block[(j, i)] * result[(offset + j, c)];
                        }
                        result[(offset + i, c)] = val / block[(i, i)];
                    }
                }
            }
            result
        }
        BlockType::Zero { rows, .. } => Mat::zeros(*rows, b.ncols()),
    }
}

fn block_logdet(block: &BlockType) -> f64 {
    match block {
        BlockType::Dense(m) => (0..m.nrows()).map(|i| m[(i, i)].ln()).sum(),
        BlockType::Diagonal(d) => d.iter().map(|x| x.ln()).sum(),
        BlockType::BlockDiagonal { block_size, blocks } => blocks
            .iter()
            .map(|b| (0..*block_size).map(|i| b[(i, i)].ln()).sum::<f64>())
            .sum(),
        BlockType::Zero { .. } => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::linalg::solvers::Solve;

    fn make_test_structures() -> Vec<RandomEffectStructure> {
        vec![
            RandomEffectStructure {
                n_levels: 3,
                n_terms: 2,
                correlated: true,
            },
            RandomEffectStructure {
                n_levels: 2,
                n_terms: 1,
                correlated: false,
            },
        ]
    }

    fn make_test_lambda_blocks() -> Vec<Mat<f64>> {
        let mut l1 = Mat::zeros(2, 2);
        l1[(0, 0)] = 1.0;
        l1[(1, 0)] = 0.3;
        l1[(1, 1)] = 0.9;

        let mut l2 = Mat::zeros(1, 1);
        l2[(0, 0)] = 0.8;

        vec![l1, l2]
    }

    fn make_test_ztwz(q: usize) -> Mat<f64> {
        let mut m = Mat::zeros(q, q);
        for i in 0..q {
            m[(i, i)] = 5.0 + i as f64;
            for j in 0..i {
                let val = 0.5 / (1.0 + (i as f64 - j as f64).abs());
                m[(i, j)] = val;
                m[(j, i)] = val;
            }
        }
        m
    }

    #[test]
    fn test_blocked_matrix_construction() {
        let structures = make_test_structures();
        let lambda_blocks = make_test_lambda_blocks();
        let q = 3 * 2 + 2;
        let ztwz = make_test_ztwz(q);

        let blocked = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, true);

        assert_eq!(blocked.block_dims.len(), 2);
        assert_eq!(blocked.block_dims[0], 6);
        assert_eq!(blocked.block_dims[1], 2);
    }

    #[test]
    fn test_blocked_cholesky_matches_dense() {
        let structures = make_test_structures();
        let lambda_blocks = make_test_lambda_blocks();
        let q = 3 * 2 + 2;
        let ztwz = make_test_ztwz(q);

        let blocked = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, true);
        let dense_v = blocked.to_dense();

        let blocked_chol = BlockedCholesky::factor(&blocked).expect("Blocked Cholesky failed");

        let dense_chol = Llt::new(dense_v.as_ref(), Side::Lower).expect("Dense Cholesky failed");

        let blocked_logdet = blocked_chol.logdet();
        let l_dense = dense_chol.L();
        let dense_logdet: f64 = 2.0 * (0..q).map(|i| l_dense[(i, i)].ln()).sum::<f64>();

        assert!(
            (blocked_logdet - dense_logdet).abs() < 1e-10,
            "logdet mismatch: blocked={}, dense={}",
            blocked_logdet,
            dense_logdet
        );
    }

    #[test]
    fn test_blocked_solve() {
        let structures = make_test_structures();
        let lambda_blocks = make_test_lambda_blocks();
        let q = 3 * 2 + 2;
        let ztwz = make_test_ztwz(q);

        let blocked = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, true);
        let dense_v = blocked.to_dense();

        let blocked_chol = BlockedCholesky::factor(&blocked).expect("Blocked Cholesky failed");
        let dense_chol = Llt::new(dense_v.as_ref(), Side::Lower).expect("Dense Cholesky failed");

        let b = Mat::from_fn(q, 1, |i, _| (i + 1) as f64);

        let blocked_x = blocked_chol.solve(&b);
        let dense_x = dense_chol.solve(&b);

        for i in 0..q {
            assert!(
                (blocked_x[(i, 0)] - dense_x[(i, 0)]).abs() < 1e-10,
                "solve mismatch at {}: blocked={}, dense={}",
                i,
                blocked_x[(i, 0)],
                dense_x[(i, 0)]
            );
        }
    }

    #[test]
    fn test_block_diagonal_chol() {
        let mut blocks = Vec::new();
        for i in 0..3 {
            let mut b = Mat::zeros(2, 2);
            b[(0, 0)] = 4.0 + i as f64;
            b[(0, 1)] = 1.0;
            b[(1, 0)] = 1.0;
            b[(1, 1)] = 4.0 + i as f64;
            blocks.push(b);
        }

        let block_diag = BlockType::BlockDiagonal {
            block_size: 2,
            blocks,
        };

        let chol = chol_block(&block_diag).expect("Cholesky failed");

        if let BlockType::BlockDiagonal {
            blocks: l_blocks, ..
        } = chol
        {
            assert_eq!(l_blocks.len(), 3);
            for l_block in &l_blocks {
                assert!(l_block[(0, 0)] > 0.0);
                assert!(l_block[(1, 1)] > 0.0);
            }
        } else {
            panic!("Expected BlockDiagonal result");
        }
    }

    #[test]
    fn test_single_block_structure() {
        let structures = vec![RandomEffectStructure {
            n_levels: 5,
            n_terms: 2,
            correlated: true,
        }];

        let mut lambda = Mat::zeros(2, 2);
        lambda[(0, 0)] = 1.5;
        lambda[(1, 0)] = 0.2;
        lambda[(1, 1)] = 1.2;
        let lambda_blocks = vec![lambda];

        let q = 10;
        let ztwz = make_test_ztwz(q);

        let blocked = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, true);
        let dense_v = blocked.to_dense();

        let blocked_chol = BlockedCholesky::factor(&blocked).expect("Blocked Cholesky failed");
        let dense_chol = Llt::new(dense_v.as_ref(), Side::Lower).expect("Dense Cholesky failed");

        let b = Mat::from_fn(q, 2, |i, j| (i + j + 1) as f64);

        let blocked_x = blocked_chol.solve(&b);
        let dense_x = dense_chol.solve(&b);

        for i in 0..q {
            for j in 0..2 {
                assert!(
                    (blocked_x[(i, j)] - dense_x[(i, j)]).abs() < 1e-9,
                    "solve mismatch at ({}, {}): blocked={}, dense={}",
                    i,
                    j,
                    blocked_x[(i, j)],
                    dense_x[(i, j)]
                );
            }
        }
    }
}
