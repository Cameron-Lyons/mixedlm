use faer::linalg::solvers::Llt;
use faer::{Mat, MatMut, MatRef, Side};
use rayon::prelude::*;

use crate::covariance::{right_apply_repeated_in_place, transpose_apply_repeated_in_place};
use crate::linalg::LinalgError;
use crate::lmm::RandomEffectStructure;

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

impl BlockType {
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
}

#[derive(Debug, Clone)]
pub struct BlockedMatrix {
    pub block_dims: Vec<usize>,
    pub blocks: Vec<Vec<BlockType>>,
}

fn independent_level_block(
    crossproducts: MatRef<'_, f64>,
    lambda: &Mat<f64>,
    levels: usize,
    add_identity: bool,
    column_stride: usize,
) -> BlockType {
    let width = lambda.nrows();
    if width == 1 {
        let scale = lambda[(0, 0)];
        let diagonal = (0..levels)
            .map(|level| {
                let mut value = (scale * crossproducts[(level, level * column_stride)]) * scale;
                if add_identity {
                    value += 1.0;
                }
                value
            })
            .collect();
        BlockType::Diagonal(diagonal)
    } else {
        let blocks = (0..levels)
            .map(|level| {
                let block =
                    crossproducts.submatrix(level * width, level * column_stride, width, width);
                let mut result = lambda.transpose() * block * lambda;
                if add_identity {
                    for i in 0..width {
                        result[(i, i)] += 1.0;
                    }
                }
                result
            })
            .collect();
        BlockType::BlockDiagonal {
            block_size: width,
            blocks,
        }
    }
}

impl BlockedMatrix {
    /// Identify structures whose weighted design products separate by level.
    /// Cache this alongside immutable crossproducts, rather than scanning each solve.
    pub fn independent_levels(ztwz: &Mat<f64>, structures: &[RandomEffectStructure]) -> Vec<bool> {
        let mut offset = 0;
        structures
            .iter()
            .map(|structure| {
                let width = structure.n_terms;
                let dimension = width * structure.n_levels;
                let independent = (0..dimension).all(|column| {
                    (0..column / width * width).all(|row| {
                        ztwz[(offset + row, offset + column)] == 0.0
                            && ztwz[(offset + column, offset + row)] == 0.0
                    })
                });
                offset += dimension;
                independent
            })
            .collect()
    }

    #[cfg(test)]
    pub fn from_lambda_ztwz(
        ztwz: &Mat<f64>,
        lambda_blocks: &[Mat<f64>],
        structures: &[RandomEffectStructure],
        add_identity: bool,
    ) -> Self {
        let independent_levels = Self::independent_levels(ztwz, structures);
        Self::from_lambda_ztwz_with_pattern(
            ztwz,
            lambda_blocks,
            structures,
            add_identity,
            &independent_levels,
        )
    }

    pub fn from_lambda_ztwz_with_pattern(
        ztwz: &Mat<f64>,
        lambda_blocks: &[Mat<f64>],
        structures: &[RandomEffectStructure],
        add_identity: bool,
        independent_levels: &[bool],
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

            for j in 0..=i {
                let qj = structures[j].n_terms;
                let nj = structures[j].n_levels;
                let lambda_j = &lambda_blocks[j];
                let offset_i = block_offsets[i];
                let offset_j = block_offsets[j];

                if i == j && independent_levels[i] {
                    row_blocks.push(independent_level_block(
                        ztwz.submatrix(offset_i, offset_i, block_dims[i], block_dims[i]),
                        lambda_i,
                        ni,
                        add_identity,
                        qi,
                    ));
                } else {
                    let mut dense_block = ztwz
                        .submatrix(offset_i, offset_j, block_dims[i], block_dims[j])
                        .to_owned();
                    transpose_apply_repeated_in_place(
                        lambda_i,
                        ni,
                        !structures[i].correlated || qi == 1,
                        dense_block.as_mut(),
                    );
                    right_apply_repeated_in_place(
                        lambda_j,
                        nj,
                        !structures[j].correlated || qj == 1,
                        dense_block.as_mut(),
                    );

                    if i == j && add_identity {
                        for diagonal in 0..block_dims[i] {
                            dense_block[(diagonal, diagonal)] += 1.0;
                        }
                    }

                    let is_zero = dense_block
                        .col_iter()
                        .all(|col| col.iter().all(|&v| v == 0.0));

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

    /// Assemble independent level blocks directly from stacked crossproducts.
    pub fn from_independent_level_crossproducts(
        crossproducts: &[Mat<f64>],
        lambda_blocks: &[Mat<f64>],
        structures: &[RandomEffectStructure],
        add_identity: bool,
    ) -> Self {
        let block_dims: Vec<_> = structures.iter().map(|s| s.n_levels * s.n_terms).collect();
        let blocks = structures
            .iter()
            .enumerate()
            .map(|(i, structure)| {
                let mut row: Vec<_> = (0..i)
                    .map(|j| BlockType::Zero {
                        rows: block_dims[i],
                        cols: block_dims[j],
                    })
                    .collect();
                row.push(independent_level_block(
                    crossproducts[i].as_ref(),
                    &lambda_blocks[i],
                    structure.n_levels,
                    add_identity,
                    0,
                ));
                row
            })
            .collect();
        Self { block_dims, blocks }
    }

    #[cfg(test)]
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
    pub fn factor(a: &BlockedMatrix) -> Result<Self, LinalgError> {
        let n_blocks = a.blocks.len();
        let block_dims = a.block_dims.clone();
        let mut l_blocks: Vec<Vec<BlockType>> = Vec::with_capacity(n_blocks);

        for (i, a_row) in a.blocks.iter().enumerate() {
            let mut row_blocks: Vec<BlockType> = Vec::with_capacity(i + 1);

            for (j, a_block) in a_row.iter().enumerate() {
                if i == j {
                    let mut aii = a_block.clone();

                    for lik in row_blocks.iter().take(i) {
                        rank_update_subtract(&mut aii, lik, lik)?;
                    }

                    let lii = chol_block(&aii)?;
                    row_blocks.push(lii);
                } else {
                    let mut aij = a_block.clone();

                    for (lik, ljk) in row_blocks.iter().take(j).zip(l_blocks[j].iter().take(j)) {
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
        self.solve_owned(b.clone())
    }

    /// Build the inverse in its final buffer, avoiding a second identity copy.
    pub fn inverse(&self) -> Mat<f64> {
        let dimension = self.block_dims.iter().sum();
        self.solve_owned(Mat::identity(dimension, dimension))
    }

    /// Invert independent levels using stacked identities with one level's width.
    /// Coupled factors require the full inverse instead.
    pub fn independent_level_inverses(&self) -> Option<Vec<Mat<f64>>> {
        let widths = self
            .l_blocks
            .iter()
            .enumerate()
            .map(|(i, row)| {
                if row[..i]
                    .iter()
                    .any(|block| !matches!(block, BlockType::Zero { .. }))
                {
                    return None;
                }
                match &row[i] {
                    BlockType::Diagonal(_) => Some(1),
                    BlockType::BlockDiagonal { block_size, .. } => Some(*block_size),
                    _ => None,
                }
            })
            .collect::<Option<Vec<_>>>()?;
        Some(
            widths
                .into_iter()
                .enumerate()
                .map(|(i, width)| {
                    let mut inverse = Mat::from_fn(self.block_dims[i], width, |row, column| {
                        if row % width == column { 1.0 } else { 0.0 }
                    });
                    solve_lower_block_in_place(&self.l_blocks[i][i], inverse.as_mut());
                    solve_lower_transpose_block_in_place(&self.l_blocks[i][i], inverse.as_mut());
                    inverse
                })
                .collect(),
        )
    }

    fn solve_owned(&self, mut rhs: Mat<f64>) -> Mat<f64> {
        self.forward_solve_in_place(&mut rhs);
        self.backward_solve_in_place(&mut rhs);
        rhs
    }

    pub fn solve_lower(&self, b: &Mat<f64>) -> Mat<f64> {
        let mut rhs = b.clone();
        self.forward_solve_in_place(&mut rhs);
        rhs
    }

    fn forward_solve_in_place(&self, rhs: &mut Mat<f64>) {
        let total_dim: usize = self.block_dims.iter().sum();
        assert_eq!(rhs.nrows(), total_dim);
        let ncols = rhs.ncols();
        let mut block_offsets = vec![0usize];
        for dim in &self.block_dims {
            block_offsets.push(block_offsets.last().unwrap() + dim);
        }

        for (i, row_blocks) in self.l_blocks.iter().enumerate() {
            let offset_i = block_offsets[i];
            let dim_i = self.block_dims[i];
            for (j, lij) in row_blocks.iter().enumerate().take(i) {
                let offset_j = block_offsets[j];
                let dim_j = self.block_dims[j];
                let contrib = block_matvec(lij, rhs.subrows(offset_j, dim_j));
                for c in 0..ncols {
                    for ii in 0..dim_i {
                        rhs[(offset_i + ii, c)] -= contrib[(ii, c)];
                    }
                }
            }
            solve_lower_block_in_place(&row_blocks[i], rhs.subrows_mut(offset_i, dim_i));
        }
    }

    fn backward_solve_in_place(&self, rhs: &mut Mat<f64>) {
        let ncols = rhs.ncols();
        let mut block_offsets = vec![0usize];
        for dim in &self.block_dims {
            block_offsets.push(block_offsets.last().unwrap() + dim);
        }

        for (i, row_blocks_i) in self.l_blocks.iter().enumerate().rev() {
            let offset_i = block_offsets[i];
            let dim_i = self.block_dims[i];
            for (j, row_blocks_j) in self.l_blocks.iter().enumerate().skip(i + 1) {
                let offset_j = block_offsets[j];
                let dim_j = self.block_dims[j];
                let contrib =
                    block_matvec_transpose(&row_blocks_j[i], rhs.subrows(offset_j, dim_j));
                for c in 0..ncols {
                    for ii in 0..dim_i {
                        rhs[(offset_i + ii, c)] -= contrib[(ii, c)];
                    }
                }
            }
            solve_lower_transpose_block_in_place(
                &row_blocks_i[i],
                rhs.subrows_mut(offset_i, dim_i),
            );
        }
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
            #[cfg(miri)]
            let results: Result<Vec<Mat<f64>>, LinalgError> = blocks
                .iter()
                .map(|b| {
                    let chol = Llt::new(b.as_ref(), Side::Lower)
                        .map_err(|_| LinalgError::NotPositiveDefinite)?;
                    Ok(chol.L().to_owned())
                })
                .collect();

            #[cfg(not(miri))]
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
        (BlockType::Diagonal(_), BlockType::Diagonal(l_diag), BlockType::Diagonal(r_diag)) => {
            if let BlockType::Diagonal(t_diag) = target {
                for i in 0..t_diag.len() {
                    t_diag[i] -= l_diag[i] * r_diag[i];
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

    let needs_conversion = match &*target {
        BlockType::BlockDiagonal { .. } => {
            matches!(l, BlockType::Dense(_)) || matches!(r, BlockType::Dense(_))
        }
        BlockType::Diagonal(_) => {
            !matches!((l, r), (BlockType::Diagonal(_), BlockType::Diagonal(_)))
        }
        _ => false,
    };

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

fn solve_lower_rows_into(l: &Mat<f64>, b: &Mat<f64>, result: &mut Mat<f64>, column_offset: usize) {
    for row in 0..b.nrows() {
        for column in 0..l.nrows() {
            let mut value = b[(row, column_offset + column)];
            for previous in 0..column {
                value -= l[(column, previous)] * result[(row, column_offset + previous)];
            }
            result[(row, column_offset + column)] = value / l[(column, column)];
        }
    }
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
                solve_lower_rows_into(l_block, b_mat, &mut result, k * bs);
            }

            Ok(BlockType::Dense(result))
        }
        (BlockType::Dense(l_mat), BlockType::Dense(b_mat)) => {
            let nrows = b_mat.nrows();
            let ncols = b_mat.ncols();
            let mut result = Mat::zeros(nrows, ncols);
            solve_lower_rows_into(l_mat, b_mat, &mut result, 0);

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
            solve_lower_rows_into(&l_dense, &b_dense, &mut result, 0);

            Ok(BlockType::Dense(result))
        }
    }
}

fn block_matvec(block: &BlockType, v: MatRef<'_, f64>) -> Mat<f64> {
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

fn block_matvec_transpose(block: &BlockType, v: MatRef<'_, f64>) -> Mat<f64> {
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

fn solve_lower_block_in_place(l: &BlockType, mut rhs: MatMut<'_, f64>) {
    match l {
        BlockType::Dense(m) => m.as_ref().solve_lower_triangular_in_place(rhs),
        BlockType::Diagonal(d) => {
            for c in 0..rhs.ncols() {
                for (i, diagonal) in d.iter().enumerate() {
                    rhs[(i, c)] /= diagonal;
                }
            }
        }
        BlockType::BlockDiagonal { block_size, blocks } => {
            // Complete each right-hand side in column-major storage order.
            for c in 0..rhs.ncols() {
                for (k, block) in blocks.iter().enumerate() {
                    let offset = k * block_size;
                    for i in 0..*block_size {
                        let mut value = rhs[(offset + i, c)];
                        for j in 0..i {
                            value -= block[(i, j)] * rhs[(offset + j, c)];
                        }
                        rhs[(offset + i, c)] = value / block[(i, i)];
                    }
                }
            }
        }
        BlockType::Zero { .. } => rhs.fill(0.0),
    }
}

fn solve_lower_transpose_block_in_place(l: &BlockType, mut rhs: MatMut<'_, f64>) {
    match l {
        BlockType::Dense(m) => m.as_ref().transpose().solve_upper_triangular_in_place(rhs),
        BlockType::Diagonal(d) => {
            for c in 0..rhs.ncols() {
                for (i, diagonal) in d.iter().enumerate() {
                    rhs[(i, c)] /= diagonal;
                }
            }
        }
        BlockType::BlockDiagonal { block_size, blocks } => {
            for c in 0..rhs.ncols() {
                for (k, block) in blocks.iter().enumerate() {
                    let offset = k * block_size;
                    for i in (0..*block_size).rev() {
                        let mut value = rhs[(offset + i, c)];
                        for j in (i + 1)..*block_size {
                            value -= block[(j, i)] * rhs[(offset + j, c)];
                        }
                        rhs[(offset + i, c)] = value / block[(i, i)];
                    }
                }
            }
        }
        BlockType::Zero { .. } => rhs.fill(0.0),
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

    #[test]
    fn stacked_level_assembly_matches_full_crossproducts() {
        let widths: &[usize] = if cfg!(miri) {
            &[1, 2]
        } else {
            &[1, 2, 3, 16, 17]
        };
        for &width in widths {
            let structures = [
                RandomEffectStructure {
                    n_levels: 3,
                    n_terms: width,
                    correlated: true,
                },
                RandomEffectStructure {
                    n_levels: 2,
                    n_terms: 2,
                    correlated: false,
                },
            ];
            let q = 3 * width + 4;
            let mut full = Mat::zeros(q, q);
            let mut stacked = Vec::new();
            let mut offset = 0;
            for structure in &structures {
                let w = structure.n_terms;
                let block = Mat::from_fn(structure.n_levels * w, w, |row, column| {
                    if row % w == column {
                        2.0 + row as f64 / 8.0
                    } else {
                        0.125
                    }
                });
                for level in 0..structure.n_levels {
                    full.submatrix_mut(offset + level * w, offset + level * w, w, w)
                        .copy_from(block.subrows(level * w, w));
                }
                offset += block.nrows();
                stacked.push(block);
            }
            for zero in [false, true] {
                let lambda: Vec<_> = structures
                    .iter()
                    .map(|structure| {
                        Mat::from_fn(structure.n_terms, structure.n_terms, |row, column| {
                            if zero || column > row || (!structure.correlated && row != column) {
                                0.0
                            } else if row == column {
                                0.5 + row as f64 / 16.0
                            } else {
                                -0.125
                            }
                        })
                    })
                    .collect();
                for identity in [false, true] {
                    let expected =
                        BlockedMatrix::from_lambda_ztwz(&full, &lambda, &structures, identity);
                    let actual = BlockedMatrix::from_independent_level_crossproducts(
                        &stacked,
                        &lambda,
                        &structures,
                        identity,
                    );
                    assert_eq!(actual.to_dense(), expected.to_dense());
                    assert!(matches!(actual.blocks[1][0], BlockType::Zero { .. }));
                }
                let factor = crate::covariance::CovarianceFactor::from_blocks(lambda, &structures);
                assert_eq!(
                    factor.right_apply_stacked_level_crossproducts(&stacked),
                    factor.right_apply_level_crossproducts(full.as_ref())
                );
            }
        }
    }

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
    fn compact_level_inverses_match_full_solves_and_reject_coupling() {
        let widths: &[usize] = if cfg!(miri) {
            &[1, 2]
        } else {
            &[1, 2, 3, 16, 17]
        };
        for &width in widths {
            let structures = [
                RandomEffectStructure {
                    n_levels: 3,
                    n_terms: width,
                    correlated: true,
                },
                RandomEffectStructure {
                    n_levels: 2,
                    n_terms: 2,
                    correlated: false,
                },
            ];
            let q = 3 * width + 4;
            let mut crossproduct = Mat::<f64>::identity(q, q);
            for level in 0..3 {
                if width > 1 {
                    crossproduct[(level * width, level * width + 1)] = 0.1;
                    crossproduct[(level * width + 1, level * width)] = 0.1;
                }
            }
            let blocks = vec![
                Mat::from_fn(width, width, |i, j| {
                    if i == j {
                        0.8
                    } else if i > j {
                        0.1 / width as f64
                    } else {
                        0.0
                    }
                }),
                Mat::from_fn(2, 2, |i, j| if i == j { 0.5 } else { 0.0 }),
            ];
            let blocked =
                BlockedMatrix::from_lambda_ztwz(&crossproduct, &blocks, &structures, true);
            let chol = BlockedCholesky::factor(&blocked).unwrap();
            let compact = chol.independent_level_inverses().unwrap();
            let full = chol.inverse();
            let dense = Llt::new(blocked.to_dense().as_ref(), Side::Lower)
                .unwrap()
                .solve(&Mat::<f64>::identity(q, q));
            let mut offset = 0;
            for (inverse, structure) in compact.iter().zip(&structures) {
                let width = structure.n_terms;
                assert_eq!(inverse.shape(), (structure.n_levels * width, width));
                for row in 0..inverse.nrows() {
                    for column in 0..width {
                        let global_column = offset + row / width * width + column;
                        assert_eq!(inverse[(row, column)], full[(offset + row, global_column)]);
                        assert!(
                            (inverse[(row, column)] - dense[(offset + row, global_column)]).abs()
                                < 1e-12
                        );
                    }
                }
                offset += inverse.nrows();
            }
            for column in [width, q - 1] {
                let mut coupled = crossproduct.clone();
                coupled[(0, column)] = 0.1;
                coupled[(column, 0)] = 0.1;
                let blocked = BlockedMatrix::from_lambda_ztwz(&coupled, &blocks, &structures, true);
                assert!(
                    BlockedCholesky::factor(&blocked)
                        .unwrap()
                        .independent_level_inverses()
                        .is_none()
                );
            }
        }
    }

    #[test]
    fn mixed_width_assembly_matches_dense_products_for_all_block_patterns() {
        let widths: &[(usize, usize)] = if cfg!(miri) {
            &[(1, 2), (3, 1)]
        } else {
            &[(1, 3), (3, 1), (15, 3), (16, 3), (17, 16), (32, 17)]
        };
        for &(left_width, right_width) in widths {
            let structures = [
                RandomEffectStructure {
                    n_levels: 2,
                    n_terms: left_width,
                    correlated: true,
                },
                RandomEffectStructure {
                    n_levels: 3,
                    n_terms: right_width,
                    correlated: false,
                },
            ];
            let split = 2 * left_width;
            let q = split + 3 * right_width;
            for independent in [false, true] {
                let ztwz = Mat::from_fn(q, q, |i, j| {
                    let cross_level = if i < split && j < split {
                        i / left_width != j / left_width
                    } else if i >= split && j >= split {
                        (i - split) / right_width != (j - split) / right_width
                    } else {
                        false
                    };
                    if i == j {
                        3.0 + (i % 5) as f64 / 4.0
                    } else if independent && cross_level {
                        0.0
                    } else {
                        0.01 / (1 + i.abs_diff(j)) as f64
                    }
                });
                for variance in 0..3 {
                    let blocks = structures
                        .iter()
                        .map(|structure| {
                            let width = structure.n_terms;
                            Mat::from_fn(width, width, |i, j| {
                                if variance == 2
                                    || i < j
                                    || (variance == 1 && j == width - 1)
                                    || (!structure.correlated && i != j)
                                {
                                    0.0
                                } else if i == j {
                                    0.4 + i as f64 / 32.0
                                } else {
                                    (i + j + 1) as f64 / 64.0
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    let mut lambda = Mat::zeros(q, q);
                    let mut offset = 0;
                    for (structure, block) in structures.iter().zip(&blocks) {
                        for _ in 0..structure.n_levels {
                            lambda
                                .submatrix_mut(offset, offset, structure.n_terms, structure.n_terms)
                                .copy_from(block.as_ref());
                            offset += structure.n_terms;
                        }
                    }
                    for identity in [false, true] {
                        let actual =
                            BlockedMatrix::from_lambda_ztwz(&ztwz, &blocks, &structures, identity);
                        if independent {
                            assert!(matches!(
                                actual.blocks[0][0],
                                BlockType::Diagonal(_) | BlockType::BlockDiagonal { .. }
                            ));
                        }
                        if variance == 2 {
                            assert!(matches!(actual.blocks[1][0], BlockType::Zero { .. }));
                        }
                        let mut expected = lambda.transpose() * &ztwz * &lambda;
                        if identity {
                            expected += Mat::<f64>::identity(q, q);
                        }
                        let dense = actual.to_dense();
                        for column in 0..q {
                            for row in 0..q {
                                assert!(
                                    (dense[(row, column)] - expected[(row, column)]).abs() < 2e-12
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn overlapping_levels_preserve_the_full_covariance_transform() {
        let structures = make_test_structures();
        let lambda_blocks = make_test_lambda_blocks();
        let ztwz = make_test_ztwz(8);
        let mut lambda = Mat::zeros(8, 8);
        let mut offset = 0;
        for (structure, block) in structures.iter().zip(&lambda_blocks) {
            for _ in 0..structure.n_levels {
                for i in 0..structure.n_terms {
                    for j in 0..structure.n_terms {
                        lambda[(offset + i, offset + j)] = block[(i, j)];
                    }
                }
                offset += structure.n_terms;
            }
        }
        assert_eq!(
            BlockedMatrix::independent_levels(&ztwz, &structures),
            [false, false]
        );
        for add_identity in [false, true] {
            let blocked =
                BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, add_identity);
            assert!(matches!(blocked.blocks[0][0], BlockType::Dense(_)));
            assert!(matches!(blocked.blocks[1][1], BlockType::Dense(_)));
            let mut expected = lambda.transpose() * &ztwz * &lambda;
            if add_identity {
                expected += Mat::<f64>::identity(8, 8);
            }
            let actual = blocked.to_dense();
            for i in 0..8 {
                for j in 0..8 {
                    assert!((actual[(i, j)] - expected[(i, j)]).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn independent_levels_keep_specialized_blocks_and_detect_small_couplings() {
        let structures = make_test_structures();
        let lambda_blocks = make_test_lambda_blocks();
        let mut ztwz = Mat::<f64>::identity(8, 8);
        ztwz[(0, 1)] = 0.2;
        ztwz[(1, 0)] = 0.2;
        ztwz[(0, 6)] = 0.3;
        ztwz[(6, 0)] = 0.3;
        assert_eq!(
            BlockedMatrix::independent_levels(&ztwz, &structures),
            [true, true]
        );
        let blocked = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, true);
        assert!(matches!(
            blocked.blocks[0][0],
            BlockType::BlockDiagonal { .. }
        ));
        assert!(matches!(blocked.blocks[1][1], BlockType::Diagonal(_)));
        ztwz[(0, 3)] = 1e-300;
        ztwz[(3, 0)] = 1e-300;
        assert_eq!(
            BlockedMatrix::independent_levels(&ztwz, &structures),
            [false, true]
        );
    }

    #[test]
    fn small_cross_structure_products_are_not_treated_as_zero() {
        let structures = make_test_structures();
        let lambda_blocks = make_test_lambda_blocks();
        let mut ztwz = Mat::<f64>::identity(8, 8);
        ztwz[(0, 6)] = 1e-300;
        ztwz[(6, 0)] = 1e-300;
        let blocked = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, true);
        assert!(matches!(blocked.blocks[1][0], BlockType::Dense(_)));
        assert!(blocked.to_dense()[(6, 0)] > 0.0);
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
    fn test_crossed_block_solve_multiple_rhs() {
        let structures = vec![
            RandomEffectStructure {
                n_levels: 4,
                n_terms: 2,
                correlated: true,
            },
            RandomEffectStructure {
                n_levels: 3,
                n_terms: 2,
                correlated: true,
            },
        ];
        let lambda_blocks = structures
            .iter()
            .map(|_| {
                let mut lambda = Mat::zeros(2, 2);
                lambda[(0, 0)] = 1.1;
                lambda[(1, 0)] = 0.2;
                lambda[(1, 1)] = 0.9;
                lambda
            })
            .collect::<Vec<_>>();
        let q = 4 * 2 + 3 * 2;
        let widths: &[usize] = if cfg!(miri) {
            &[0, 1, 4]
        } else {
            &[0, 1, 4, 17, 129]
        };
        for independent in [false, true] {
            let mut ztwz = make_test_ztwz(q);
            if independent {
                for i in 0..q {
                    for j in 0..q {
                        if (i < 8) == (j < 8) && i / 2 != j / 2 {
                            ztwz[(i, j)] = 0.0;
                        }
                    }
                }
            }
            let blocked = BlockedMatrix::from_lambda_ztwz(&ztwz, &lambda_blocks, &structures, true);
            let dense_v = blocked.to_dense();
            let blocked_chol = BlockedCholesky::factor(&blocked).unwrap();
            let dense_chol = Llt::new(dense_v.as_ref(), Side::Lower).unwrap();
            let inverse = blocked_chol.inverse();
            let expected_inverse = dense_chol.solve(&Mat::<f64>::identity(q, q));
            for row in 0..q {
                for column in 0..q {
                    assert!(
                        (inverse[(row, column)] - expected_inverse[(row, column)]).abs() < 1e-12
                    );
                }
            }
            for &width in widths {
                let b = Mat::from_fn(q, width, |row, column| (row + 2 * column + 1) as f64);
                let original = b.clone();
                let blocked_x = blocked_chol.solve(&b);
                let dense_x = dense_chol.solve(&b);
                let blocked_lower = blocked_chol.solve_lower(&b);
                let mut dense_lower = b.clone();
                dense_chol
                    .L()
                    .solve_lower_triangular_in_place(dense_lower.as_mut());
                assert_eq!(b, original);
                assert_eq!(blocked_x.ncols(), width);
                assert_eq!(blocked_lower.ncols(), width);
                for row in 0..q {
                    for column in 0..width {
                        assert!((blocked_x[(row, column)] - dense_x[(row, column)]).abs() < 1e-10);
                        assert!(
                            (blocked_lower[(row, column)] - dense_lower[(row, column)]).abs()
                                < 1e-10
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn triangular_blocks_solve_strided_views_without_touching_adjacent_rows() {
        let q = 9;
        let lower = |n| {
            Mat::from_fn(n, n, |i, j| {
                if i == j {
                    3.0 + i as f64 / 8.0
                } else if j < i {
                    (i + j + 1) as f64 / 16.0
                } else {
                    0.0
                }
            })
        };
        let factors = [
            BlockType::Dense(lower(q)),
            BlockType::Diagonal(vec![3.0; q]),
            BlockType::BlockDiagonal {
                block_size: 3,
                blocks: vec![lower(3); 3],
            },
            BlockType::Zero { rows: q, cols: q },
        ];
        let widths: &[usize] = if cfg!(miri) {
            &[0, 1, 4]
        } else {
            &[0, 1, 4, 17, 129]
        };
        for factor in &factors {
            for &width in widths {
                let b = Mat::from_fn(q, width, |i, j| ((3 * i + 7 * j) % 17) as f64 / 9.0 - 1.0);
                for transpose in [false, true] {
                    let mut buffer = Mat::from_fn(q + 2, width, |_, _| 42.0);
                    buffer.subrows_mut(1, q).copy_from(b.as_ref());
                    if transpose {
                        solve_lower_transpose_block_in_place(factor, buffer.subrows_mut(1, q));
                    } else {
                        solve_lower_block_in_place(factor, buffer.subrows_mut(1, q));
                    }
                    let dense = factor.to_dense();
                    let actual = if transpose {
                        dense.transpose() * buffer.subrows(1, q)
                    } else {
                        &dense * buffer.subrows(1, q)
                    };
                    for column in 0..width {
                        assert_eq!(buffer[(0, column)], 42.0);
                        assert_eq!(buffer[(q + 1, column)], 42.0);
                        for row in 0..q {
                            if matches!(factor, BlockType::Zero { .. }) {
                                assert_eq!(buffer[(row + 1, column)], 0.0);
                            } else {
                                assert!((actual[(row, column)] - b[(row, column)]).abs() < 1e-12);
                            }
                        }
                    }
                }
            }
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
