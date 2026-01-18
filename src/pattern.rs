use core::fmt;
use core::ops::Range;

use faer_core::sparse::SymbolicSparseColMatRef;

/// Column-compressed sparsity pattern for a Jacobian J(x).
///
/// Indices are zero-based; each column's row indices must be sorted.
#[derive(Debug, Clone)]
pub struct JacobianPattern {
    nrows: usize,
    ncols: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
}

/// Validation errors for a JacobianPattern.
#[derive(Debug, Clone)]
pub enum PatternError {
    /// col_ptrs length is not ncols + 1.
    ColPtrLen { expected: usize, actual: usize },
    /// col_ptrs[0] is not 0.
    ColPtrStart { value: usize },
    /// col_ptrs is not non-decreasing.
    ColPtrNotMonotonic { col: usize, prev: usize, next: usize },
    /// col_ptrs[ncols] does not match row_indices length.
    ColPtrOutOfBounds { last: usize, row_indices_len: usize },
    /// A row index is >= nrows.
    RowIndexOutOfBounds { col: usize, row: usize, nrows: usize },
    /// Row indices in a column are not sorted.
    RowIndexNotSorted { col: usize, prev: usize, next: usize },
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColPtrLen { expected, actual } => {
                write!(f, "col_ptrs length {actual} does not match expected {expected}")
            }
            Self::ColPtrStart { value } => {
                write!(f, "col_ptrs must start at 0 (got {value})")
            }
            Self::ColPtrNotMonotonic { col, prev, next } => {
                write!(f, "col_ptrs not monotonic at col {col}: {prev} > {next}")
            }
            Self::ColPtrOutOfBounds {
                last,
                row_indices_len,
            } => {
                write!(
                    f,
                    "col_ptrs end {last} exceeds row_indices length {row_indices_len}"
                )
            }
            Self::RowIndexOutOfBounds { col, row, nrows } => {
                write!(
                    f,
                    "row index {row} in col {col} exceeds nrows {nrows}"
                )
            }
            Self::RowIndexNotSorted { col, prev, next } => {
                write!(
                    f,
                    "row indices not sorted in col {col}: {prev} > {next}"
                )
            }
        }
    }
}

impl std::error::Error for PatternError {}

impl JacobianPattern {
    /// Creates a validated column-compressed sparsity pattern.
    ///
    /// Requirements:
    /// - `col_ptrs.len() == ncols + 1`
    /// - `col_ptrs` is non-decreasing and starts at `0`
    /// - `col_ptrs[ncols] == row_indices.len()`
    /// - row indices are sorted and `< nrows` within each column
    pub fn new(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
    ) -> Result<Self, PatternError> {
        let expected = ncols + 1;
        if col_ptrs.len() != expected {
            return Err(PatternError::ColPtrLen {
                expected,
                actual: col_ptrs.len(),
            });
        }
        if col_ptrs.first().copied().unwrap_or(0) != 0 {
            return Err(PatternError::ColPtrStart {
                value: col_ptrs[0],
            });
        }
        for col in 0..ncols {
            let prev = col_ptrs[col];
            let next = col_ptrs[col + 1];
            if prev > next {
                return Err(PatternError::ColPtrNotMonotonic { col, prev, next });
            }
        }
        let last = col_ptrs[ncols];
        if last != row_indices.len() {
            return Err(PatternError::ColPtrOutOfBounds {
                last,
                row_indices_len: row_indices.len(),
            });
        }

        for col in 0..ncols {
            let start = col_ptrs[col];
            let end = col_ptrs[col + 1];
            if start == end {
                continue;
            }
            let mut prev = row_indices[start];
            if prev >= nrows {
                return Err(PatternError::RowIndexOutOfBounds {
                    col,
                    row: prev,
                    nrows,
                });
            }
            for &row in &row_indices[start + 1..end] {
                if prev >= row {
                    return Err(PatternError::RowIndexNotSorted { col, prev, next: row });
                }
                if row >= nrows {
                    return Err(PatternError::RowIndexOutOfBounds {
                        col,
                        row,
                        nrows,
                    });
                }
                prev = row;
            }
        }

        Ok(Self {
            nrows,
            ncols,
            col_ptrs,
            row_indices,
        })
    }

    /// Number of residuals (rows in J).
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of parameters (columns in J).
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of non-zeros in J.
    pub fn nnz(&self) -> usize {
        self.row_indices.len()
    }

    /// Column pointer array in CSC format.
    pub fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }

    /// Row index array in CSC format.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Index range in row_indices for the given column.
    pub fn col_range(&self, col: usize) -> Range<usize> {
        self.col_ptrs[col]..self.col_ptrs[col + 1]
    }

    /// Sorted row indices for the given column.
    pub fn row_indices_of_col(&self, col: usize) -> &[usize] {
        let range = self.col_range(col);
        &self.row_indices[range]
    }
}

#[derive(Debug)]
pub(crate) struct AugmentedPattern {
    jacobian_rows: usize,
    ncols: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    diag_positions: Vec<usize>,
}

impl AugmentedPattern {
    pub(crate) fn new(jacobian: &JacobianPattern) -> Self {
        let ncols = jacobian.ncols();
        let jacobian_rows = jacobian.nrows();
        let mut col_ptrs = Vec::with_capacity(ncols + 1);
        col_ptrs.push(0);
        for col in 0..ncols {
            let nnz = jacobian.col_ptrs[col + 1] - jacobian.col_ptrs[col];
            let next = col_ptrs[col] + nnz + 1;
            col_ptrs.push(next);
        }

        let mut row_indices = Vec::with_capacity(jacobian.nnz() + ncols);
        let mut diag_positions = Vec::with_capacity(ncols);
        for col in 0..ncols {
            let range = jacobian.col_range(col);
            row_indices.extend_from_slice(&jacobian.row_indices[range]);
            row_indices.push(jacobian_rows + col);
            diag_positions.push(row_indices.len() - 1);
        }

        Self {
            jacobian_rows,
            ncols,
            col_ptrs,
            row_indices,
            diag_positions,
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.jacobian_rows + self.ncols
    }

    pub(crate) fn ncols(&self) -> usize {
        self.ncols
    }

    pub(crate) fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }

    pub(crate) fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    pub(crate) fn diag_positions(&self) -> &[usize] {
        &self.diag_positions
    }

    pub(crate) fn jacobian_rows(&self) -> usize {
        self.jacobian_rows
    }

    pub(crate) fn as_symbolic(&self) -> SymbolicSparseColMatRef<'_, usize> {
        unsafe {
            SymbolicSparseColMatRef::new_unchecked(
                self.nrows(),
                self.ncols,
                &self.col_ptrs,
                None,
                &self.row_indices,
            )
        }
    }
}
