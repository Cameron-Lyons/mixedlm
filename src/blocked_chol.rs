//! Cholesky factors of the random-effect precision V = Lambda' Z'WZ Lambda + I.
//!
//! V is held as symmetric level tiles. `LevelCholesky` factors each level's
//! tile when no level couples another. `BlockedCholesky` eliminates leading
//! mutually independent levels one tile at a time and the remaining levels as
//! a dense Schur complement. `SparseCholesky` analyses the fixed tile pattern
//! once for an AMD-ordered simplicial LDL' factorization. All can select the
//! inverse on the tile pattern, which the covariance gradient contracts with.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::sparse::SymbolicSparseColMatRef;
use faer::sparse::linalg::amd;
use faer::{Mat, MatMut, MatRef, Par};

use crate::csc::{LevelTiles, SCALAR_WIDTH};
use crate::linalg::LinalgError;

/// Dense Schur complements at least this wide use the global thread pool.
/// Smaller ones and per-level tiles finish faster than parallel dispatch.
const PARALLEL_DIMENSION: usize = 256;

fn parallelism(dimension: usize) -> Par {
    if dimension >= PARALLEL_DIMENSION {
        faer::get_global_parallelism()
    } else {
        Par::Seq
    }
}

/// Factor a column-major tile in place, reading and writing its lower triangle.
fn cholesky_tile(tile: &mut [f64], width: usize) -> Result<(), LinalgError> {
    if width >= SCALAR_WIDTH {
        return cholesky_dense(MatMut::from_column_major_slice_mut(tile, width, width));
    }
    for j in 0..width {
        let mut pivot = tile[j + j * width];
        for k in 0..j {
            pivot -= tile[j + k * width] * tile[j + k * width];
        }
        if !(pivot > 0.0 && pivot.is_finite()) {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let pivot = pivot.sqrt();
        tile[j + j * width] = pivot;
        for i in j + 1..width {
            let mut value = tile[i + j * width];
            for k in 0..j {
                value -= tile[i + k * width] * tile[j + k * width];
            }
            tile[i + j * width] = value / pivot;
        }
    }
    Ok(())
}

fn cholesky_dense(mut matrix: MatMut<'_, f64>) -> Result<(), LinalgError> {
    let dimension = matrix.nrows();
    let par = parallelism(dimension);
    let mut scratch = MemBuffer::new(cholesky_in_place_scratch::<f64>(
        dimension,
        par,
        Default::default(),
    ));
    cholesky_in_place(
        matrix.as_mut(),
        Default::default(),
        par,
        MemStack::new(&mut scratch),
        Default::default(),
    )
    .map_err(|_| LinalgError::NotPositiveDefinite)?;
    if (0..dimension).any(|i| !(matrix[(i, i)] > 0.0 && matrix[(i, i)].is_finite())) {
        return Err(LinalgError::NotPositiveDefinite);
    }
    Ok(())
}

/// Write the symmetric inverse of a dense lower Cholesky factor.
fn dense_inverse(lower: MatRef<'_, f64>, mut output: MatMut<'_, f64>) {
    let dimension = lower.nrows();
    let par = parallelism(dimension);
    let mut scratch = MemBuffer::new(llt::inverse::inverse_scratch::<f64>(dimension, par));
    // L^-1 and then the lower triangle of L^-T L^-1, about m^3 / 3 operations.
    llt::inverse::inverse(output.as_mut(), lower, par, MemStack::new(&mut scratch));
    for j in 1..dimension {
        for i in 0..j {
            output[(i, j)] = output[(j, i)];
        }
    }
}

/// Write the inverse L^-T L^-1 of a column-major factor tile, using `scratch`
/// for L^-1.
fn tile_inverse(lower: &[f64], width: usize, scratch: &mut Vec<f64>, output: &mut [f64]) {
    if width >= SCALAR_WIDTH {
        dense_inverse(
            MatRef::from_column_major_slice(lower, width, width),
            MatMut::from_column_major_slice_mut(output, width, width),
        );
        return;
    }
    let inverse_factor = scratch;
    inverse_factor.clear();
    inverse_factor.resize(width * width, 0.0);
    for j in 0..width {
        inverse_factor[j + j * width] = 1.0;
        solve_tile_lower(
            lower,
            width,
            &mut inverse_factor[j * width..(j + 1) * width],
        );
    }
    for j in 0..width {
        for i in 0..width {
            output[i + j * width] = (i.max(j)..width)
                .map(|k| inverse_factor[k + i * width] * inverse_factor[k + j * width])
                .sum();
        }
    }
}

/// Replace a column-major tile by tile * L^-T for a lower factor tile L.
fn solve_right_transpose(lower: &[f64], width: usize, tile: &mut [f64]) {
    let height = tile.len() / width;
    for j in 0..width {
        for k in 0..j {
            let coefficient = lower[j + k * width];
            for i in 0..height {
                tile[i + j * height] -= tile[i + k * height] * coefficient;
            }
        }
        let pivot = lower[j + j * width];
        for value in &mut tile[j * height..(j + 1) * height] {
            *value /= pivot;
        }
    }
}

/// Replace a column-major tile by tile * L^-1 for a lower factor tile L.
fn solve_right(lower: &[f64], width: usize, tile: &mut [f64]) {
    let height = tile.len() / width;
    for j in (0..width).rev() {
        for k in j + 1..width {
            let coefficient = lower[k + j * width];
            for i in 0..height {
                tile[i + j * height] -= tile[i + k * height] * coefficient;
            }
        }
        let pivot = lower[j + j * width];
        for value in &mut tile[j * height..(j + 1) * height] {
            *value /= pivot;
        }
    }
}

/// Solve L x = b in place for a lower factor tile.
fn solve_tile_lower(lower: &[f64], width: usize, rhs: &mut [f64]) {
    for j in 0..width {
        rhs[j] /= lower[j + j * width];
        for i in j + 1..width {
            rhs[i] -= lower[i + j * width] * rhs[j];
        }
    }
}

/// Solve L' x = b in place for a lower factor tile.
fn solve_tile_upper(lower: &[f64], width: usize, rhs: &mut [f64]) {
    for j in (0..width).rev() {
        for i in j + 1..width {
            rhs[j] -= lower[i + j * width] * rhs[i];
        }
        rhs[j] /= lower[j + j * width];
    }
}

/// Lower Cholesky factor of V when no level couples another: one factor tile
/// per level, laid out as the tiles' structure runs.
pub struct LevelCholesky<'a> {
    tiles: &'a LevelTiles,
    values: Vec<f64>,
}

impl<'a> LevelCholesky<'a> {
    /// Factor values on block-diagonal `tiles` in place.
    pub fn factor(tiles: &'a LevelTiles, mut values: Vec<f64>) -> Result<Self, LinalgError> {
        assert!(tiles.block_diagonal() && values.len() == tiles.len());
        for (width, span, _) in tiles.runs() {
            let values = &mut values[span];
            if width == 1 {
                for value in values {
                    if !(*value > 0.0 && value.is_finite()) {
                        return Err(LinalgError::NotPositiveDefinite);
                    }
                    *value = value.sqrt();
                }
                continue;
            }
            for tile in values.chunks_exact_mut(width * width) {
                cholesky_tile(tile, width)?;
            }
        }
        Ok(Self { tiles, values })
    }

    /// Solve L x = b, or L' x = b when `TRANSPOSE`, in place.
    pub fn solve_in_place<const TRANSPOSE: bool>(&self, rhs: &mut Mat<f64>) {
        for column in 0..rhs.ncols() {
            let rhs = rhs.col_as_slice_mut(column);
            for (width, span, rows) in self.tiles.runs() {
                let (lower, rhs) = (&self.values[span], &mut rhs[rows]);
                if width == 1 {
                    for (value, pivot) in rhs.iter_mut().zip(lower) {
                        *value /= pivot;
                    }
                    continue;
                }
                for (lower, rhs) in lower
                    .chunks_exact(width * width)
                    .zip(rhs.chunks_exact_mut(width))
                {
                    if TRANSPOSE {
                        solve_tile_upper(lower, width, rhs);
                    } else {
                        solve_tile_lower(lower, width, rhs);
                    }
                }
            }
        }
    }

    pub fn logdet(&self) -> f64 {
        let logdet: f64 = self
            .tiles
            .runs()
            .map(|(width, span, _)| {
                let lower = &self.values[span];
                if width == 1 {
                    return lower.iter().map(|pivot| pivot.ln()).sum::<f64>();
                }
                lower
                    .chunks_exact(width * width)
                    .map(|lower| (0..width).map(|i| lower[i * (width + 1)].ln()).sum::<f64>())
                    .sum()
            })
            .sum();
        2.0 * logdet
    }

    /// V^-1, whose tiles are the inverses of the levels' tiles.
    pub fn selected_inverse(&self) -> Vec<f64> {
        let mut inverse = vec![0.0; self.values.len()];
        let mut scratch = Vec::new();
        for (width, span, _) in self.tiles.runs() {
            let (lower, output) = (&self.values[span.clone()], &mut inverse[span]);
            if width == 1 {
                for (output, pivot) in output.iter_mut().zip(lower) {
                    let inverse_pivot = 1.0 / pivot;
                    *output = inverse_pivot * inverse_pivot;
                }
                continue;
            }
            let size = width * width;
            for (lower, output) in lower.chunks_exact(size).zip(output.chunks_exact_mut(size)) {
                tile_inverse(lower, width, &mut scratch, output);
            }
        }
        inverse
    }
}

/// Lower Cholesky factor of V. Leading levels have diagonal factor tiles L_c
/// and couplings W_rc = V_rc L_c^-T below them. The Schur complement of the
/// trailing levels, V_tt - W W', is factored densely.
pub struct BlockedCholesky<'a> {
    tiles: &'a LevelTiles,
    leading: usize,
    values: Vec<f64>,
    trailing: Mat<f64>,
}

impl<'a> BlockedCholesky<'a> {
    /// Factor values on `tiles`' pattern in place. The first `leading` levels
    /// must be mutually independent, as given by `LevelTiles::independent_prefix`.
    pub fn factor(
        tiles: &'a LevelTiles,
        leading: usize,
        mut values: Vec<f64>,
    ) -> Result<Self, LinalgError> {
        assert_eq!(values.len(), tiles.len());
        let first = tiles.start(leading);
        let dimension = tiles.dimension() - first;
        for level in 0..leading {
            let width = tiles.width(level);
            let column = tiles.column(level);
            let diagonal = tiles.span(column.start);
            let (head, tail) = values.split_at_mut(diagonal.end);
            let lower = &mut head[diagonal.clone()];
            cholesky_tile(lower, width)?;
            for tile in column.skip(1) {
                let span = tiles.span(tile);
                solve_right_transpose(
                    lower,
                    width,
                    &mut tail[span.start - diagonal.end..span.end - diagonal.end],
                );
            }
        }
        let mut trailing = Mat::zeros(dimension, dimension);
        for level in leading..tiles.n_levels() {
            let left = tiles.start(level) - first;
            for tile in tiles.column(level) {
                let row = tiles.row(tile);
                let (top, height) = (tiles.start(row) - first, tiles.width(row));
                for (index, &value) in values[tiles.span(tile)].iter().enumerate() {
                    let (i, j) = (index % height, index / height);
                    if row != level || i >= j {
                        trailing[(top + i, left + j)] = value;
                    }
                }
            }
        }
        // Each leading level updates the trailing levels it couples, W_c W_c'.
        for level in 0..leading {
            let width = tiles.width(level);
            let couplings = tiles.column(level).skip(1);
            for (a, tile_a) in couplings.clone().enumerate() {
                let row_a = tiles.row(tile_a);
                let (top, height_a) = (tiles.start(row_a) - first, tiles.width(row_a));
                let coupling_a = &values[tiles.span(tile_a)];
                for tile_b in couplings.clone().take(a + 1) {
                    let row_b = tiles.row(tile_b);
                    let (left, height_b) = (tiles.start(row_b) - first, tiles.width(row_b));
                    let coupling_b = &values[tiles.span(tile_b)];
                    for j in 0..height_b {
                        let rows = if row_a == row_b {
                            j..height_a
                        } else {
                            0..height_a
                        };
                        for i in rows {
                            let mut value = 0.0;
                            for k in 0..width {
                                value +=
                                    coupling_a[i + k * height_a] * coupling_b[j + k * height_b];
                            }
                            trailing[(top + i, left + j)] -= value;
                        }
                    }
                }
            }
        }
        if dimension > 0 {
            cholesky_dense(trailing.as_mut())?;
        }
        Ok(Self {
            tiles,
            leading,
            values,
            trailing,
        })
    }

    /// Solve L x = b in place.
    pub fn solve_lower_in_place(&self, rhs: &mut Mat<f64>) {
        let tiles = self.tiles;
        let first = tiles.start(self.leading);
        for column in 0..rhs.ncols() {
            let rhs = rhs.col_as_slice_mut(column);
            for level in 0..self.leading {
                let (start, width) = (tiles.start(level), tiles.width(level));
                let mut tiles_in_column = tiles.column(level);
                let lower = &self.values[tiles.span(tiles_in_column.next().unwrap())];
                let (head, tail) = rhs.split_at_mut(start + width);
                let solved = &mut head[start..];
                solve_tile_lower(lower, width, solved);
                for tile in tiles_in_column {
                    let row = tiles.row(tile);
                    let (top, height) = (tiles.start(row) - start - width, tiles.width(row));
                    let coupling = &self.values[tiles.span(tile)];
                    for (k, &value) in solved.iter().enumerate() {
                        for i in 0..height {
                            tail[top + i] -= coupling[i + k * height] * value;
                        }
                    }
                }
            }
        }
        if self.trailing.nrows() > 0 {
            solve_lower_triangular_in_place(
                self.trailing.as_ref(),
                rhs.as_mut().subrows_mut(first, self.trailing.nrows()),
                parallelism(self.trailing.nrows()),
            );
        }
    }

    /// Solve L' x = b in place.
    pub fn solve_upper_in_place(&self, rhs: &mut Mat<f64>) {
        let tiles = self.tiles;
        let first = tiles.start(self.leading);
        if self.trailing.nrows() > 0 {
            solve_upper_triangular_in_place(
                self.trailing.transpose(),
                rhs.as_mut().subrows_mut(first, self.trailing.nrows()),
                parallelism(self.trailing.nrows()),
            );
        }
        for column in 0..rhs.ncols() {
            let rhs = rhs.col_as_slice_mut(column);
            for level in 0..self.leading {
                let (start, width) = (tiles.start(level), tiles.width(level));
                let mut tiles_in_column = tiles.column(level);
                let lower = &self.values[tiles.span(tiles_in_column.next().unwrap())];
                let (head, tail) = rhs.split_at_mut(start + width);
                let solved = &mut head[start..];
                for tile in tiles_in_column {
                    let row = tiles.row(tile);
                    let (top, height) = (tiles.start(row) - start - width, tiles.width(row));
                    let coupling = &self.values[tiles.span(tile)];
                    for (k, value) in solved.iter_mut().enumerate() {
                        for i in 0..height {
                            *value -= coupling[i + k * height] * tail[top + i];
                        }
                    }
                }
                solve_tile_upper(lower, width, solved);
            }
        }
    }

    pub fn logdet(&self) -> f64 {
        let tiles = self.tiles;
        let leading: f64 = (0..self.leading)
            .map(|level| {
                let width = tiles.width(level);
                let lower = &self.values[tiles.span(tiles.column(level).start)];
                (0..width).map(|i| lower[i * (width + 1)].ln()).sum::<f64>()
            })
            .sum();
        let trailing: f64 = (0..self.trailing.nrows())
            .map(|i| self.trailing[(i, i)].ln())
            .sum();
        2.0 * (leading + trailing)
    }

    /// V^-1 on the tile pattern. With U = W L^-1 for each leading level,
    /// V^-1 couples it to trailing rows as -Z U and to itself as
    /// L^-T L^-1 + U' Z U, where Z is the dense inverse over trailing levels.
    pub fn selected_inverse(&self) -> Vec<f64> {
        let tiles = self.tiles;
        let first = tiles.start(self.leading);
        let dimension = self.trailing.nrows();
        let mut trailing = Mat::zeros(dimension, dimension);
        dense_inverse(self.trailing.as_ref(), trailing.as_mut());
        let mut inverse = vec![0.0; tiles.len()];
        for level in self.leading..tiles.n_levels() {
            let left = tiles.start(level) - first;
            for tile in tiles.column(level) {
                let row = tiles.row(tile);
                let (top, height) = (tiles.start(row) - first, tiles.width(row));
                for (index, value) in inverse[tiles.span(tile)].iter_mut().enumerate() {
                    *value = trailing[(top + index % height, left + index / height)];
                }
            }
        }
        let (mut products, mut scratch, mut block) = (Vec::new(), Vec::new(), Vec::new());
        for level in 0..self.leading {
            let width = tiles.width(level);
            let mut column = tiles.column(level);
            let diagonal = tiles.span(column.next().unwrap());
            let lower = &self.values[diagonal.clone()];
            // U tiles, stacked in the order of the couplings.
            products.clear();
            for tile in column.clone() {
                let start = products.len();
                products.extend_from_slice(&self.values[tiles.span(tile)]);
                solve_right(lower, width, &mut products[start..]);
            }
            for tile_a in column.clone() {
                let row_a = tiles.row(tile_a);
                let (top, height_a) = (tiles.start(row_a) - first, tiles.width(row_a));
                let output = &mut inverse[tiles.span(tile_a)];
                let mut offset = 0;
                for tile_b in column.clone() {
                    let row_b = tiles.row(tile_b);
                    let (left, height_b) = (tiles.start(row_b) - first, tiles.width(row_b));
                    let product = &products[offset..offset + height_b * width];
                    for j in 0..width {
                        for i in 0..height_a {
                            let mut value = 0.0;
                            for k in 0..height_b {
                                value += trailing[(top + i, left + k)] * product[k + j * height_b];
                            }
                            output[i + j * height_a] -= value;
                        }
                    }
                    offset += height_b * width;
                }
            }
            // L^-T L^-1, then add U' Z U = -U' (V^-1 below).
            block.resize(width * width, 0.0);
            tile_inverse(lower, width, &mut scratch, &mut block);
            let mut offset = 0;
            for tile in column {
                let height = tiles.width(tiles.row(tile));
                let product = &products[offset..offset + height * width];
                let below = &inverse[tiles.span(tile)];
                for j in 0..width {
                    for i in 0..width {
                        block[i + j * width] -= (0..height)
                            .map(|k| product[k + i * height] * below[k + j * height])
                            .sum::<f64>();
                    }
                }
                offset += height * width;
            }
            inverse[diagonal].copy_from_slice(&block);
        }
        inverse
    }
}

/// Analysis of a fixed tile pattern for an AMD-ordered simplicial LDL'
/// factorization. Factors keep L, so the inverse can be selected on its pattern.
#[derive(Debug)]
pub struct SparseCholesky {
    /// Original index of each eliminated position.
    permutation: Vec<usize>,
    /// Columns of the permuted upper triangle, with each entry's tile value.
    upper_offsets: Vec<usize>,
    upper_rows: Vec<usize>,
    upper_sources: Vec<usize>,
    /// Elimination tree and the strictly lower pattern of L with ascending rows.
    parent: Vec<usize>,
    factor_offsets: Vec<usize>,
    factor_rows: Vec<usize>,
    /// Position of each tile value in L's pattern, or after it on the diagonal.
    inverse_sources: Vec<usize>,
}

impl SparseCholesky {
    /// Analyse the pattern, or return None if the factorization's work, taken
    /// as the sum of squared column counts of L, would exceed `max_flops`.
    pub fn new(tiles: &LevelTiles, max_flops: usize) -> Option<Self> {
        let n = tiles.dimension();
        // The lower pattern in original order, with each entry's tile value.
        let (mut offsets, mut rows, mut sources) = (vec![0], Vec::new(), Vec::new());
        for level in 0..tiles.n_levels() {
            for local in 0..tiles.width(level) {
                for tile in tiles.column(level) {
                    let row = tiles.row(tile);
                    let (top, height) = (tiles.start(row), tiles.width(row));
                    let span = tiles.span(tile);
                    for i in if row == level { local } else { 0 }..height {
                        rows.push(top + i);
                        sources.push(span.start + i + local * height);
                    }
                }
                offsets.push(rows.len());
            }
        }
        let mut permutation = vec![0; n];
        let mut inverse = vec![0; n];
        if n > 0 {
            let mut memory = MemBuffer::new(amd::order_scratch::<usize>(n, rows.len()));
            // Zero the workspace before faer's AMD, as sparse_chol.rs does.
            for byte in memory.iter_mut() {
                byte.write(0);
            }
            amd::order(
                &mut permutation,
                &mut inverse,
                SymbolicSparseColMatRef::new_checked(n, n, &offsets, None, &rows),
                Default::default(),
                MemStack::new(&mut memory),
            )
            .ok()?;
        }
        let mut upper_offsets = vec![0; n + 1];
        for column in 0..n {
            for &row in &rows[offsets[column]..offsets[column + 1]] {
                upper_offsets[inverse[row].max(inverse[column]) + 1] += 1;
            }
        }
        for column in 0..n {
            upper_offsets[column + 1] += upper_offsets[column];
        }
        let mut cursors = upper_offsets[..n].to_vec();
        let mut upper_rows = vec![0; rows.len()];
        let mut upper_sources = vec![0; rows.len()];
        for column in 0..n {
            for position in offsets[column]..offsets[column + 1] {
                let (i, j) = (inverse[rows[position]], inverse[column]);
                let target = &mut cursors[i.max(j)];
                upper_rows[*target] = i.min(j);
                upper_sources[*target] = sources[position];
                *target += 1;
            }
        }
        // Each row k of L reaches up the elimination tree from row k of the
        // upper triangle. Count those entries, then record them in row order.
        let mut parent = vec![usize::MAX; n];
        let mut flags = vec![usize::MAX; n];
        let mut factor_offsets = vec![0usize; n + 1];
        for k in 0..n {
            flags[k] = k;
            for &row in &upper_rows[upper_offsets[k]..upper_offsets[k + 1]] {
                let mut i = row;
                while flags[i] != k {
                    if parent[i] == usize::MAX {
                        parent[i] = k;
                    }
                    factor_offsets[i + 1] += 1;
                    flags[i] = k;
                    i = parent[i];
                }
            }
        }
        let flops = factor_offsets.iter().fold(0usize, |total, &count| {
            total.saturating_add(count.saturating_mul(count))
        });
        if flops > max_flops {
            return None;
        }
        for column in 0..n {
            factor_offsets[column + 1] += factor_offsets[column];
        }
        let mut cursors = factor_offsets[..n].to_vec();
        let mut factor_rows = vec![0; factor_offsets[n]];
        flags.fill(usize::MAX);
        for k in 0..n {
            flags[k] = k;
            for &row in &upper_rows[upper_offsets[k]..upper_offsets[k + 1]] {
                let mut i = row;
                while flags[i] != k {
                    factor_rows[cursors[i]] = k;
                    cursors[i] += 1;
                    flags[i] = k;
                    i = parent[i];
                }
            }
        }
        let mut inverse_sources = vec![0; tiles.len()];
        for level in 0..tiles.n_levels() {
            let left = tiles.start(level);
            for tile in tiles.column(level) {
                let (top, height) = (tiles.start(tiles.row(tile)), tiles.width(tiles.row(tile)));
                for (index, position) in tiles.span(tile).enumerate() {
                    let i = inverse[top + index % height];
                    let j = inverse[left + index / height];
                    inverse_sources[position] = if i == j {
                        factor_rows.len() + i
                    } else {
                        let (row, column) = (i.max(j), i.min(j));
                        let entries = factor_offsets[column]..factor_offsets[column + 1];
                        entries.start
                            + factor_rows[entries]
                                .binary_search(&row)
                                .expect("the filled pattern contains the original pattern")
                    };
                }
            }
        }
        Some(Self {
            permutation,
            upper_offsets,
            upper_rows,
            upper_sources,
            parent,
            factor_offsets,
            factor_rows,
            inverse_sources,
        })
    }

    /// Factor values on the analysed tile pattern.
    pub fn factor(&self, values: &[f64]) -> Result<SparseLdl<'_>, LinalgError> {
        let n = self.parent.len();
        let mut lower = vec![0.0; self.factor_rows.len()];
        let mut diagonal = vec![0.0; n];
        let mut work = vec![0.0; n];
        let mut flags = vec![usize::MAX; n];
        let mut pattern = vec![0; n];
        let mut next = self.factor_offsets[..n].to_vec();
        for k in 0..n {
            // Scatter row k and find the columns it updates in topological order.
            flags[k] = k;
            let mut top = n;
            for position in self.upper_offsets[k]..self.upper_offsets[k + 1] {
                let mut i = self.upper_rows[position];
                work[i] += values[self.upper_sources[position]];
                let mut length = 0;
                while flags[i] != k {
                    pattern[length] = i;
                    length += 1;
                    flags[i] = k;
                    i = self.parent[i];
                }
                while length > 0 {
                    top -= 1;
                    length -= 1;
                    pattern[top] = pattern[length];
                }
            }
            let mut pivot = work[k];
            work[k] = 0.0;
            for &i in &pattern[top..] {
                let value = work[i];
                work[i] = 0.0;
                for position in self.factor_offsets[i]..next[i] {
                    work[self.factor_rows[position]] -= lower[position] * value;
                }
                let entry = value / diagonal[i];
                pivot -= entry * value;
                lower[next[i]] = entry;
                next[i] += 1;
            }
            if !(pivot > 0.0 && pivot.is_finite()) {
                return Err(LinalgError::NotPositiveDefinite);
            }
            diagonal[k] = pivot;
        }
        Ok(SparseLdl {
            analysis: self,
            lower,
            diagonal,
        })
    }
}

/// Numeric LDL' factor of P V P' on an analysed pattern.
pub struct SparseLdl<'a> {
    analysis: &'a SparseCholesky,
    lower: Vec<f64>,
    diagonal: Vec<f64>,
}

impl SparseLdl<'_> {
    fn forward(&self, work: &mut [f64]) {
        let analysis = self.analysis;
        for j in 0..work.len() {
            let value = work[j];
            for position in analysis.factor_offsets[j]..analysis.factor_offsets[j + 1] {
                work[analysis.factor_rows[position]] -= self.lower[position] * value;
            }
        }
    }

    /// Whiten each column by D^-1/2 L^-1 P, so its crossproducts are b' V^-1 b.
    /// Rows of the result are in elimination order.
    pub fn solve_lower_in_place(&self, rhs: &mut Mat<f64>) {
        let mut work = vec![0.0; self.diagonal.len()];
        for column in 0..rhs.ncols() {
            let rhs = rhs.col_as_slice_mut(column);
            for (value, &source) in work.iter_mut().zip(&self.analysis.permutation) {
                *value = rhs[source];
            }
            self.forward(&mut work);
            for ((output, value), pivot) in rhs.iter_mut().zip(&work).zip(&self.diagonal) {
                *output = value / pivot.sqrt();
            }
        }
    }

    /// Solve V x = b in place.
    pub fn solve_in_place(&self, rhs: &mut Mat<f64>) {
        let analysis = self.analysis;
        let mut work = vec![0.0; self.diagonal.len()];
        for column in 0..rhs.ncols() {
            let rhs = rhs.col_as_slice_mut(column);
            for (value, &source) in work.iter_mut().zip(&analysis.permutation) {
                *value = rhs[source];
            }
            self.forward(&mut work);
            for (value, pivot) in work.iter_mut().zip(&self.diagonal) {
                *value /= pivot;
            }
            for j in (0..work.len()).rev() {
                let mut value = work[j];
                for position in analysis.factor_offsets[j]..analysis.factor_offsets[j + 1] {
                    value -= self.lower[position] * work[analysis.factor_rows[position]];
                }
                work[j] = value;
            }
            for (&value, &target) in work.iter().zip(&analysis.permutation) {
                rhs[target] = value;
            }
        }
    }

    pub fn logdet(&self) -> f64 {
        self.diagonal.iter().map(|pivot| pivot.ln()).sum()
    }

    /// V^-1 on the tile pattern, by Takahashi's recurrences on the pattern of L:
    /// Z_ij = -sum_k Z_ik L_kj and Z_jj = 1/D_j - sum_k L_kj Z_kj over rows k of
    /// column j. Those rows form a clique, so every Z_ik lies in L's pattern.
    pub fn selected_inverse(&self) -> Vec<f64> {
        let analysis = self.analysis;
        let (offsets, rows) = (&analysis.factor_offsets, &analysis.factor_rows);
        let entries = rows.len();
        let mut inverse = vec![0.0; entries + self.diagonal.len()];
        let mut column = Vec::new();
        for j in (0..self.diagonal.len()).rev() {
            let span = offsets[j]..offsets[j + 1];
            let (rows_j, lower_j) = (&rows[span.clone()], &self.lower[span.clone()]);
            column.clear();
            column.resize(rows_j.len(), 0.0);
            for (b, (&row_b, &lower_b)) in rows_j.iter().zip(lower_j).enumerate() {
                column[b] += inverse[entries + row_b] * lower_b;
                // Merge the later rows of column j with the sorted rows of column row_b.
                let mut position = offsets[row_b];
                for a in b + 1..rows_j.len() {
                    while rows[position] != rows_j[a] {
                        position += 1;
                    }
                    debug_assert!(position < offsets[row_b + 1]);
                    let value = inverse[position];
                    column[a] += value * lower_b;
                    column[b] += value * lower_j[a];
                }
            }
            let mut diagonal = 1.0 / self.diagonal[j];
            for ((output, &value), &lower) in inverse[span].iter_mut().zip(&column).zip(lower_j) {
                *output = -value;
                diagonal += value * lower;
            }
            inverse[entries + j] = diagonal;
        }
        analysis
            .inverse_sources
            .iter()
            .map(|&position| inverse[position])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Side;
    use faer::linalg::solvers::{Llt, Solve};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-10 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    /// A diagonally dominant matrix whose levels couple where `coupled` says,
    /// given the higher level first.
    fn precision(
        blocks: &[(usize, usize)],
        coupled: impl Fn(usize, usize) -> bool,
    ) -> (LevelTiles, Mat<f64>) {
        let mut level_of = Vec::new();
        for &(count, width) in blocks {
            for _ in 0..count {
                let level = level_of.last().map_or(0, |&level| level + 1);
                level_of.extend(std::iter::repeat_n(level, width));
            }
        }
        let q = level_of.len();
        let linked = |i: usize, j: usize| {
            let (high, low) = (level_of[i].max(level_of[j]), level_of[i].min(level_of[j]));
            high == low || coupled(high, low)
        };
        let mut dense = Mat::from_fn(q, q, |i, j| {
            if linked(i, j) && i != j {
                ((3 * (i + j) + i * j) % 7) as f64 / 9.0 - 0.3
            } else {
                0.0
            }
        });
        for i in 0..q {
            dense[(i, i)] = 1.0 + (0..q).map(|j| dense[(i, j)].abs()).sum::<f64>() + i as f64 / 8.0;
        }
        let tiles = LevelTiles::lower_from_entries(blocks, |i, j| dense[(i, j)]);
        (tiles, dense)
    }

    fn cases() -> Vec<(LevelTiles, Mat<f64>, usize)> {
        let nested = |high: usize, low: usize| {
            (low < 6 && high == 6 + low / 3) || (high == 8 && (low == 6 || low == 7))
        };
        let crossed =
            |high: usize, low: usize| high >= 5 && low < 5 && !(low + 2 * high).is_multiple_of(3);
        let mut cases = Vec::new();
        for (blocks, coupled, leading) in [
            (
                vec![(3, 2), (2, 1), (1, 17)],
                &(|_, _| false) as &dyn Fn(usize, usize) -> bool,
                6,
            ),
            (vec![(6, 1), (2, 2), (1, 3)], &nested, 6),
            (vec![(5, 2), (4, 1)], &crossed, 5),
            (vec![(2, 2), (1, 3)], &|_, _| true, 1),
            (vec![(1, 20), (2, 1)], &|_, _| true, 1),
            (vec![], &|_, _| true, 0),
        ] {
            let (tiles, dense) = precision(&blocks, coupled);
            assert_eq!(tiles.independent_prefix(), leading);
            cases.push((tiles, dense, leading));
        }
        cases
    }

    /// Compare a factorization's solves, log determinant and selected inverse
    /// with dense Cholesky.
    fn check_factor(
        tiles: &LevelTiles,
        dense: &Mat<f64>,
        logdet: f64,
        solve: impl Fn(&mut Mat<f64>),
        whiten: impl Fn(&mut Mat<f64>),
        selected: Vec<f64>,
    ) {
        let q = dense.nrows();
        let reference = Llt::new(dense.as_ref(), Side::Lower).unwrap();
        let expected_logdet = 2.0 * (0..q).map(|i| reference.L()[(i, i)].ln()).sum::<f64>();
        assert_close(logdet, expected_logdet);
        let rhs = Mat::from_fn(q, 3, |i, j| ((2 * i + 5 * j) % 7) as f64 - 3.0);
        let expected = reference.solve(&rhs);
        let mut actual = rhs.clone();
        solve(&mut actual);
        let mut whitened = rhs.clone();
        whiten(&mut whitened);
        let crossproduct = whitened.transpose() * &whitened;
        let expected_crossproduct = rhs.transpose() * &expected;
        for j in 0..rhs.ncols() {
            for i in 0..q {
                assert_close(actual[(i, j)], expected[(i, j)]);
            }
            for i in 0..rhs.ncols() {
                assert_close(crossproduct[(i, j)], expected_crossproduct[(i, j)]);
            }
        }
        let inverse = reference.solve(Mat::<f64>::identity(q, q));
        let pattern = tiles.dense_from(&vec![1.0; tiles.len()]);
        let selected = tiles.dense_from(&selected);
        for j in 0..q {
            for i in 0..q {
                let expected = if pattern[(i, j)] == 1.0 {
                    inverse[(i, j)]
                } else {
                    0.0
                };
                assert_close(selected[(i, j)], expected);
            }
        }
    }

    #[test]
    fn blocked_factors_match_dense_cholesky() {
        for (tiles, dense, leading) in cases() {
            let factor = BlockedCholesky::factor(&tiles, leading, tiles.values().to_vec()).unwrap();
            assert_eq!(
                factor.trailing.nrows(),
                dense.nrows() - tiles.start(leading)
            );
            check_factor(
                &tiles,
                &dense,
                factor.logdet(),
                |rhs| {
                    factor.solve_lower_in_place(rhs);
                    factor.solve_upper_in_place(rhs);
                },
                |rhs| factor.solve_lower_in_place(rhs),
                factor.selected_inverse(),
            );
        }
    }

    #[test]
    fn level_factors_match_dense_cholesky() {
        for (tiles, dense, _) in cases().iter().filter(|(tiles, ..)| tiles.block_diagonal()) {
            let factor = LevelCholesky::factor(tiles, tiles.values().to_vec()).unwrap();
            check_factor(
                tiles,
                dense,
                factor.logdet(),
                |rhs| {
                    factor.solve_in_place::<false>(rhs);
                    factor.solve_in_place::<true>(rhs);
                },
                |rhs| factor.solve_in_place::<false>(rhs),
                factor.selected_inverse(),
            );
        }
    }

    #[test]
    fn sparse_factors_match_dense_cholesky_within_their_work_budget() {
        for (tiles, dense, leading) in cases() {
            let analysis = SparseCholesky::new(&tiles, usize::MAX).unwrap();
            let factor = analysis.factor(tiles.values()).unwrap();
            check_factor(
                &tiles,
                &dense,
                factor.logdet(),
                |rhs| factor.solve_in_place(rhs),
                |rhs| factor.solve_lower_in_place(rhs),
                factor.selected_inverse(),
            );
            // A zero budget admits only factors without off-diagonal entries.
            assert_eq!(
                SparseCholesky::new(&tiles, 0).is_none(),
                !analysis.factor_rows.is_empty()
            );
            assert_eq!(leading == tiles.n_levels(), tiles.block_diagonal());
        }
    }

    #[test]
    fn nested_levels_factor_without_fill() {
        let (tiles, _, _) = &cases()[1];
        let analysis = SparseCholesky::new(tiles, usize::MAX).unwrap();
        let strictly_lower = (0..tiles.n_levels())
            .map(|level| {
                let width = tiles.width(level);
                tiles
                    .column(level)
                    .map(|tile| {
                        let height = tiles.width(tiles.row(tile));
                        if tiles.row(tile) == level {
                            width * (width - 1) / 2
                        } else {
                            height * width
                        }
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();
        assert_eq!(analysis.factor_rows.len(), strictly_lower);
    }

    #[test]
    fn factors_reject_nonpositive_and_nonfinite_pivots() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            for width in [1, 2, 17] {
                let values: Vec<f64> = (0..2 * width * width)
                    .map(|index| {
                        let (level, local) = (index / (width * width), index % (width * width));
                        match (level, local % (width + 1) == 0) {
                            (1, true) if local == (width * width - 1) => invalid,
                            (_, true) => 4.0,
                            _ => 0.0,
                        }
                    })
                    .collect();
                let tiles = LevelTiles::lower_from_entries(&[(2, width)], |i, j| {
                    let (level, row, column) = (i / width, i % width, j % width);
                    if i / width != j / width {
                        0.0
                    } else {
                        values[level * width * width + row + column * width]
                    }
                });
                assert!(matches!(
                    BlockedCholesky::factor(&tiles, 2, tiles.values().to_vec()),
                    Err(LinalgError::NotPositiveDefinite)
                ));
                assert!(matches!(
                    LevelCholesky::factor(&tiles, tiles.values().to_vec()),
                    Err(LinalgError::NotPositiveDefinite)
                ));
                let analysis = SparseCholesky::new(&tiles, usize::MAX).unwrap();
                assert!(matches!(
                    analysis.factor(tiles.values()),
                    Err(LinalgError::NotPositiveDefinite)
                ));
            }
        }
        // Each level is positive, but their Schur complement is not.
        let tiles =
            LevelTiles::lower_from_entries(&[(2, 1)], |i, j| if i == j { 1.0 } else { 2.0 });
        assert!(matches!(
            BlockedCholesky::factor(&tiles, 1, tiles.values().to_vec()),
            Err(LinalgError::NotPositiveDefinite)
        ));
        let analysis = SparseCholesky::new(&tiles, usize::MAX).unwrap();
        assert!(analysis.factor(tiles.values()).is_err());
    }
}
