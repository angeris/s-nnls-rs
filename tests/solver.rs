use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use faer_core::Parallelism;
use s_nnls_rs::{
    JacobianPattern, JacobianValuesMut, LmSolver, Problem, SolveStatus, SolverOptions,
};

struct CountingAlloc;

static ALLOC_TOTAL: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOC_TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            ALLOC_TOTAL.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            ALLOC_TOTAL.fetch_add(new_size, Ordering::Relaxed);
        }
        new_ptr
    }
}

fn reset_alloc_counter() {
    ALLOC_TOTAL.store(0, Ordering::SeqCst);
}

fn allocated_bytes() -> usize {
    ALLOC_TOTAL.load(Ordering::SeqCst)
}

struct FnProblem<R, J> {
    res: R,
    jac: J,
}

impl<R, J> Problem for FnProblem<R, J>
where
    R: FnMut(&[f64], &mut [f64]),
    J: FnMut(&[f64], &mut JacobianValuesMut<'_>),
{
    fn residuals(&mut self, x: &[f64], residuals: &mut [f64]) {
        (self.res)(x, residuals);
    }

    fn jacobian(&mut self, x: &[f64], jacobian: &mut JacobianValuesMut<'_>) {
        (self.jac)(x, jacobian);
    }
}

fn solver_options() -> SolverOptions {
    SolverOptions {
        max_iters: 200,
        verbose: false,
        ..SolverOptions::default()
    }
}

fn assert_converged(status: SolveStatus) {
    assert!(
        matches!(
            status,
            SolveStatus::ConvergedGradient | SolveStatus::ConvergedStep | SolveStatus::ConvergedCost
        ),
        "unexpected status: {status:?}"
    );
}

fn pattern_from_triplets_1b(
    nrows: usize,
    ncols: usize,
    entries: &[(usize, usize)],
) -> JacobianPattern {
    let mut cols: Vec<Vec<usize>> = vec![Vec::new(); ncols];
    for &(row, col) in entries {
        cols[col - 1].push(row - 1);
    }
    for col_rows in &mut cols {
        col_rows.sort_unstable();
        col_rows.dedup();
    }
    let mut col_ptrs = Vec::with_capacity(ncols + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col_rows in cols {
        row_indices.extend_from_slice(&col_rows);
        col_ptrs.push(row_indices.len());
    }
    JacobianPattern::new(nrows, ncols, col_ptrs, row_indices).unwrap()
}

fn zero_jacobian(jac: &mut JacobianValuesMut<'_>) {
    for col in 0..jac.ncols() {
        jac.values_of_col_mut(col).fill(0.0);
    }
}

fn set_entry_1b(jac: &mut JacobianValuesMut<'_>, row: usize, col: usize, value: f64) {
    let row0 = row - 1;
    let col0 = col - 1;
    let pos = {
        let rows = jac.row_indices_of_col(col0);
        rows.iter()
            .position(|&r| r == row0)
            .unwrap_or_else(|| panic!("missing ({row},{col}) in jacobian pattern"))
    };
    let values = jac.values_of_col_mut(col0);
    values[pos] = value;
}

fn solve_problem<R, J>(
    pattern: JacobianPattern,
    x0: Vec<f64>,
    res: R,
    jac: J,
) -> (Vec<f64>, SolveStatus)
where
    R: FnMut(&[f64], &mut [f64]),
    J: FnMut(&[f64], &mut JacobianValuesMut<'_>),
{
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut x = x0;
    let mut problem = FnProblem { res, jac };
    let stats = solver
        .solve(&mut problem, &mut x, &solver_options(), None)
        .unwrap();
    (x, stats.status)
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64)
    }
}

fn dense_pattern_entries_1b(nrows: usize, ncols: usize) -> Vec<(usize, usize)> {
    let mut entries = Vec::with_capacity(nrows * ncols);
    for col in 1..=ncols {
        for row in 1..=nrows {
            entries.push((row, col));
        }
    }
    entries
}

fn chain_pattern_entries_1b(n: usize) -> Vec<(usize, usize)> {
    let mut entries = Vec::with_capacity(3 * n);
    for col in 1..=n {
        entries.push((col, col));
        if col > 1 {
            entries.push((n + col - 1, col));
        }
        if col < n {
            entries.push((n + col, col));
        }
    }
    entries
}

struct SparseLinearProblem {
    col_rows: Vec<Vec<usize>>,
    col_vals: Vec<Vec<f64>>,
    b: Vec<f64>,
}

impl Problem for SparseLinearProblem {
    fn residuals(&mut self, x: &[f64], residuals: &mut [f64]) {
        residuals.copy_from_slice(&self.b);
        for v in residuals.iter_mut() {
            *v = -*v;
        }
        for (col, rows) in self.col_rows.iter().enumerate() {
            let xj = x[col];
            let vals = &self.col_vals[col];
            for (idx, &row) in rows.iter().enumerate() {
                residuals[row] += vals[idx] * xj;
            }
        }
    }

    fn jacobian(&mut self, _x: &[f64], jacobian: &mut JacobianValuesMut<'_>) {
        for col in 0..jacobian.ncols() {
            jacobian
                .values_of_col_mut(col)
                .copy_from_slice(&self.col_vals[col]);
        }
    }
}

fn make_sparse_linear_problem(
    ncols: usize,
    extra_rows: usize,
    nnz_per_col: usize,
    seed: u64,
) -> (JacobianPattern, SparseLinearProblem, Vec<f64>) {
    let nrows = ncols + extra_rows;
    assert!(nnz_per_col > 0);
    assert!(nnz_per_col <= nrows);
    let mut rng = Lcg::new(seed);

    let mut col_rows = Vec::with_capacity(ncols);
    let mut col_vals = Vec::with_capacity(ncols);
    for col in 0..ncols {
        let mut rows = Vec::with_capacity(nnz_per_col);
        rows.push(col);
        while rows.len() < nnz_per_col {
            let row = rng.next_usize(nrows);
            if !rows.contains(&row) {
                rows.push(row);
            }
        }
        rows.sort_unstable();
        let mut vals = Vec::with_capacity(rows.len());
        for _ in 0..rows.len() {
            let mut v = rng.next_f64() * 2.0 - 1.0;
            if v == 0.0 {
                v = 0.1;
            }
            vals.push(v);
        }
        col_rows.push(rows);
        col_vals.push(vals);
    }

    let mut x_star = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        x_star.push(rng.next_f64() * 2.0 - 1.0);
    }

    let mut b = vec![0.0; nrows];
    for col in 0..ncols {
        let xj = x_star[col];
        for (idx, &row) in col_rows[col].iter().enumerate() {
            b[row] += col_vals[col][idx] * xj;
        }
    }

    let mut entries = Vec::new();
    for col in 0..ncols {
        for &row in &col_rows[col] {
            entries.push((row + 1, col + 1));
        }
    }
    let pattern = pattern_from_triplets_1b(nrows, ncols, &entries);
    let problem = SparseLinearProblem {
        col_rows,
        col_vals,
        b,
    };
    (pattern, problem, x_star)
}

// Solves 1D linear and cubic residuals to verify basic convergence.
#[test]
fn basic_nonlinear_solves() {
    let pattern = pattern_from_triplets_1b(1, 1, &[(1, 1)]);
    let (x, status) = solve_problem(
        pattern,
        vec![0.0],
        |x, out| {
            out[0] = x[0] - 1.0;
        },
        |_x, jac| {
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
        },
    );
    assert_converged(status);
    assert!((x[0] - 1.0).abs() <= 1e-6);

    let pattern = pattern_from_triplets_1b(1, 1, &[(1, 1)]);
    let (x, status) = solve_problem(
        pattern,
        vec![0.5],
        |x, out| {
            out[0] = x[0] * x[0] * x[0] - 1.0;
        },
        |x, jac| {
            let x1 = x[0];
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 3.0 * x1 * x1);
        },
    );
    assert_converged(status);
    assert!((x[0] - 1.0).abs() <= 1e-6);
}

// Solves a coupled quadratic system with a sparse Jacobian.
#[test]
fn multidimensional_quadratics() {
    let pattern = pattern_from_triplets_1b(3, 2, &[(1, 1), (3, 1), (2, 2), (3, 2)]);
    let (x, status) = solve_problem(
        pattern,
        vec![0.8, 1.7],
        |x, out| {
            let x1 = x[0];
            let x2 = x[1];
            out[0] = x1 * x1 - 1.0;
            out[1] = x2 * x2 - 4.0;
            out[2] = x1 * x2 - 2.0;
        },
        |x, jac| {
            let x1 = x[0];
            let x2 = x[1];
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 2.0 * x1);
            set_entry_1b(jac, 3, 1, x2);
            set_entry_1b(jac, 2, 2, 2.0 * x2);
            set_entry_1b(jac, 3, 2, x1);
        },
    );
    assert_converged(status);
    assert!((x[0] - 1.0).abs() <= 1e-6);
    assert!((x[1] - 2.0).abs() <= 1e-6);
}

// Solves the classic Rosenbrock residual formulation.
#[test]
fn rosenbrock_residuals() {
    let pattern = pattern_from_triplets_1b(2, 2, &[(1, 1), (2, 1), (2, 2)]);
    let (x, status) = solve_problem(
        pattern,
        vec![-1.2, 1.0],
        |x, out| {
            let x1 = x[0];
            let x2 = x[1];
            out[0] = 1.0 - x1;
            out[1] = 10.0 * (x2 - x1 * x1);
        },
        |x, jac| {
            let x1 = x[0];
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, -1.0);
            set_entry_1b(jac, 2, 1, -20.0 * x1);
            set_entry_1b(jac, 2, 2, 10.0);
        },
    );
    assert_converged(status);
    assert!((x[0] - 1.0).abs() <= 1e-5);
    assert!((x[1] - 1.0).abs() <= 1e-5);
}

// Solves a CAD-style axis-distance constraint set.
#[test]
fn cad_axis_distance() {
    let pattern = pattern_from_triplets_1b(
        4,
        4,
        &[(1, 1), (4, 1), (2, 2), (4, 2), (4, 3), (3, 4), (4, 4)],
    );
    let (x, status) = solve_problem(
        pattern,
        vec![0.8, 1.2, 2.6, 0.1],
        |x, out| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            out[0] = x1 - 1.0;
            out[1] = y1 - 1.0;
            out[2] = y2;
            out[3] = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) - 4.0;
        },
        |x, jac| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
            set_entry_1b(jac, 4, 1, 2.0 * (x1 - x2));
            set_entry_1b(jac, 2, 2, 1.0);
            set_entry_1b(jac, 4, 2, 2.0 * (y1 - y2));
            set_entry_1b(jac, 4, 3, 2.0 * (x2 - x1));
            set_entry_1b(jac, 3, 4, 1.0);
            set_entry_1b(jac, 4, 4, 2.0 * (y2 - y1));
        },
    );
    let x2_expected = 1.0 + 3.0_f64.sqrt();
    assert_converged(status);
    assert!((x[0] - 1.0).abs() <= 1e-5);
    assert!((x[1] - 1.0).abs() <= 1e-5);
    assert!((x[2] - x2_expected).abs() <= 1e-5);
    assert!(x[3].abs() <= 1e-5);
}

// Solves CAD constraints that enforce parallel lines.
#[test]
fn cad_parallel_lines() {
    let pattern = pattern_from_triplets_1b(
        8,
        8,
        &[
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
        ],
    );
    let (x, status) = solve_problem(
        pattern,
        vec![0.1, -0.1, 2.1, 0.2, -0.1, 1.1, 1.9, 1.2],
        |x, out| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            let x3 = x[4];
            let y3 = x[5];
            let x4 = x[6];
            let y4 = x[7];
            out[0] = x1;
            out[1] = y1;
            out[2] = x2 - 2.0;
            out[3] = y2;
            out[4] = x3;
            out[5] = y3 - 1.0;
            out[6] = (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3) - 4.0;
            out[7] = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3);
        },
        |x, jac| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            let x3 = x[4];
            let y3 = x[5];
            let x4 = x[6];
            let y4 = x[7];
            let dx12 = x2 - x1;
            let dy12 = y2 - y1;
            let dx34 = x4 - x3;
            let dy34 = y4 - y3;
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
            set_entry_1b(jac, 8, 1, -dy34);
            set_entry_1b(jac, 2, 2, 1.0);
            set_entry_1b(jac, 8, 2, dx34);
            set_entry_1b(jac, 3, 3, 1.0);
            set_entry_1b(jac, 8, 3, dy34);
            set_entry_1b(jac, 4, 4, 1.0);
            set_entry_1b(jac, 8, 4, -dx34);
            set_entry_1b(jac, 5, 5, 1.0);
            set_entry_1b(jac, 7, 5, -2.0 * dx34);
            set_entry_1b(jac, 8, 5, dy12);
            set_entry_1b(jac, 6, 6, 1.0);
            set_entry_1b(jac, 7, 6, -2.0 * dy34);
            set_entry_1b(jac, 8, 6, -dx12);
            set_entry_1b(jac, 7, 7, 2.0 * dx34);
            set_entry_1b(jac, 8, 7, -dy12);
            set_entry_1b(jac, 7, 8, 2.0 * dy34);
            set_entry_1b(jac, 8, 8, dx12);
        },
    );
    assert_converged(status);
    assert!(x[0].abs() <= 1e-5);
    assert!(x[1].abs() <= 1e-5);
    assert!((x[2] - 2.0).abs() <= 1e-5);
    assert!(x[3].abs() <= 1e-5);
    assert!(x[4].abs() <= 1e-5);
    assert!((x[5] - 1.0).abs() <= 1e-5);
    assert!((x[6] - 2.0).abs() <= 1e-5);
    assert!((x[7] - 1.0).abs() <= 1e-5);
}

// Solves CAD constraints that enforce perpendicular lines.
#[test]
fn cad_perpendicular_lines() {
    let pattern = pattern_from_triplets_1b(
        8,
        8,
        &[
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 8),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 6),
            (8, 7),
            (8, 8),
        ],
    );
    let (x, status) = solve_problem(
        pattern,
        vec![0.1, -0.1, 2.1, 0.1, 1.1, 0.1, 0.9, 2.1],
        |x, out| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            let x3 = x[4];
            let y3 = x[5];
            let x4 = x[6];
            let y4 = x[7];
            out[0] = x1;
            out[1] = y1;
            out[2] = x2 - 2.0;
            out[3] = y2;
            out[4] = x3 - 1.0;
            out[5] = y3;
            out[6] = y4 - 2.0;
            out[7] = (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3);
        },
        |x, jac| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            let x3 = x[4];
            let y3 = x[5];
            let x4 = x[6];
            let y4 = x[7];
            let dx12 = x2 - x1;
            let dy12 = y2 - y1;
            let dx34 = x4 - x3;
            let dy34 = y4 - y3;
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
            set_entry_1b(jac, 8, 1, -dx34);
            set_entry_1b(jac, 2, 2, 1.0);
            set_entry_1b(jac, 8, 2, -dy34);
            set_entry_1b(jac, 3, 3, 1.0);
            set_entry_1b(jac, 8, 3, dx34);
            set_entry_1b(jac, 4, 4, 1.0);
            set_entry_1b(jac, 8, 4, dy34);
            set_entry_1b(jac, 5, 5, 1.0);
            set_entry_1b(jac, 8, 5, -dx12);
            set_entry_1b(jac, 6, 6, 1.0);
            set_entry_1b(jac, 8, 6, -dy12);
            set_entry_1b(jac, 8, 7, dx12);
            set_entry_1b(jac, 7, 8, 1.0);
            set_entry_1b(jac, 8, 8, dy12);
        },
    );
    assert_converged(status);
    assert!((x[6] - 1.0).abs() <= 1e-5);
    assert!((x[7] - 2.0).abs() <= 1e-5);
}

// Solves CAD constraints for a tangent circle.
#[test]
fn cad_tangent_circle() {
    let pattern = pattern_from_triplets_1b(
        9,
        9,
        &[
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (6, 7),
            (7, 5),
            (7, 6),
            (7, 7),
            (7, 8),
            (7, 9),
            (8, 8),
            (9, 9),
        ],
    );
    let (x, status) = solve_problem(
        pattern,
        vec![0.1, -0.1, 2.1, 0.1, 0.9, 0.8, 0.9, 2.1, 0.9],
        |x, out| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            let cx = x[4];
            let cy = x[5];
            let r = x[6];
            let x3 = x[7];
            let y3 = x[8];
            out[0] = x1;
            out[1] = y1;
            out[2] = x2 - 2.0;
            out[3] = y2;
            out[4] = cx - 1.0;
            out[5] = cy - r;
            out[6] = (x3 - cx) * (x3 - cx) + (y3 - cy) * (y3 - cy) - r * r;
            out[7] = x3 - 2.0;
            out[8] = y3 - 1.0;
        },
        |x, jac| {
            let cx = x[4];
            let cy = x[5];
            let r = x[6];
            let x3 = x[7];
            let y3 = x[8];
            let dx = x3 - cx;
            let dy = y3 - cy;
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
            set_entry_1b(jac, 2, 2, 1.0);
            set_entry_1b(jac, 3, 3, 1.0);
            set_entry_1b(jac, 4, 4, 1.0);
            set_entry_1b(jac, 5, 5, 1.0);
            set_entry_1b(jac, 6, 6, 1.0);
            set_entry_1b(jac, 6, 7, -1.0);
            set_entry_1b(jac, 7, 5, -2.0 * dx);
            set_entry_1b(jac, 7, 6, -2.0 * dy);
            set_entry_1b(jac, 7, 7, -2.0 * r);
            set_entry_1b(jac, 7, 8, 2.0 * dx);
            set_entry_1b(jac, 7, 9, 2.0 * dy);
            set_entry_1b(jac, 8, 8, 1.0);
            set_entry_1b(jac, 9, 9, 1.0);
        },
    );
    assert_converged(status);
    assert!((x[4] - 1.0).abs() <= 1e-5);
    assert!((x[5] - 1.0).abs() <= 1e-5);
    assert!((x[6] - 1.0).abs() <= 1e-5);
}

// Solves a larger CAD-style constraint bundle.
#[test]
fn cad_complex_constraints() {
    let pattern = pattern_from_triplets_1b(
        16,
        15,
        &[
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 1),
            (7, 9),
            (8, 2),
            (8, 10),
            (9, 5),
            (9, 11),
            (10, 6),
            (10, 12),
            (11, 1),
            (11, 2),
            (11, 3),
            (11, 4),
            (11, 5),
            (11, 6),
            (11, 7),
            (11, 8),
            (12, 5),
            (12, 6),
            (12, 7),
            (12, 8),
            (13, 13),
            (14, 14),
            (15, 14),
            (15, 15),
            (16, 7),
            (16, 8),
            (16, 13),
            (16, 14),
            (16, 15),
        ],
    );
    let (x, status) = solve_problem(
        pattern,
        vec![
            0.1, -0.1, 2.1, 0.2, -0.1, 1.1, 1.9, 0.9, 0.2, 0.1, -0.1, 1.2, 0.9, 1.2, 1.1,
        ],
        |x, out| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            let x3 = x[4];
            let y3 = x[5];
            let x4 = x[6];
            let y4 = x[7];
            let x5 = x[8];
            let y5 = x[9];
            let x6 = x[10];
            let y6 = x[11];
            let cx = x[12];
            let cy = x[13];
            let r = x[14];
            out[0] = x1;
            out[1] = y1;
            out[2] = x2 - 2.0;
            out[3] = y2;
            out[4] = x3;
            out[5] = y3 - 1.0;
            out[6] = x5 - x1;
            out[7] = y5 - y1;
            out[8] = x6 - x3;
            out[9] = y6 - y3;
            out[10] = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3);
            out[11] = (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3) - 4.0;
            out[12] = cx - 1.0;
            out[13] = cy - 1.0;
            out[14] = cy - r;
            out[15] = (x4 - cx) * (x4 - cx) + (y4 - cy) * (y4 - cy) - r * r;
        },
        |x, jac| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            let x3 = x[4];
            let y3 = x[5];
            let x4 = x[6];
            let y4 = x[7];
            let cx = x[12];
            let cy = x[13];
            let r = x[14];
            let dx12 = x2 - x1;
            let dy12 = y2 - y1;
            let dx34 = x4 - x3;
            let dy34 = y4 - y3;
            let dx4 = x4 - cx;
            let dy4 = y4 - cy;
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
            set_entry_1b(jac, 2, 2, 1.0);
            set_entry_1b(jac, 3, 3, 1.0);
            set_entry_1b(jac, 4, 4, 1.0);
            set_entry_1b(jac, 5, 5, 1.0);
            set_entry_1b(jac, 6, 6, 1.0);
            set_entry_1b(jac, 7, 1, -1.0);
            set_entry_1b(jac, 7, 9, 1.0);
            set_entry_1b(jac, 8, 2, -1.0);
            set_entry_1b(jac, 8, 10, 1.0);
            set_entry_1b(jac, 9, 5, -1.0);
            set_entry_1b(jac, 9, 11, 1.0);
            set_entry_1b(jac, 10, 6, -1.0);
            set_entry_1b(jac, 10, 12, 1.0);
            set_entry_1b(jac, 11, 1, -dy34);
            set_entry_1b(jac, 11, 2, dx34);
            set_entry_1b(jac, 11, 3, dy34);
            set_entry_1b(jac, 11, 4, -dx34);
            set_entry_1b(jac, 11, 5, dy12);
            set_entry_1b(jac, 11, 6, -dx12);
            set_entry_1b(jac, 11, 7, -dy12);
            set_entry_1b(jac, 11, 8, dx12);
            set_entry_1b(jac, 12, 5, -2.0 * dx34);
            set_entry_1b(jac, 12, 6, -2.0 * dy34);
            set_entry_1b(jac, 12, 7, 2.0 * dx34);
            set_entry_1b(jac, 12, 8, 2.0 * dy34);
            set_entry_1b(jac, 13, 13, 1.0);
            set_entry_1b(jac, 14, 14, 1.0);
            set_entry_1b(jac, 15, 14, 1.0);
            set_entry_1b(jac, 15, 15, -1.0);
            set_entry_1b(jac, 16, 7, 2.0 * dx4);
            set_entry_1b(jac, 16, 8, 2.0 * dy4);
            set_entry_1b(jac, 16, 13, -2.0 * dx4);
            set_entry_1b(jac, 16, 14, -2.0 * dy4);
            set_entry_1b(jac, 16, 15, -2.0 * r);
        },
    );
    assert_converged(status);
    assert!(x[0].abs() <= 1e-5);
    assert!(x[1].abs() <= 1e-5);
    assert!((x[2] - 2.0).abs() <= 1e-5);
    assert!(x[3].abs() <= 1e-5);
    assert!(x[4].abs() <= 1e-5);
    assert!((x[5] - 1.0).abs() <= 1e-5);
    assert!((x[6] - 2.0).abs() <= 1e-5);
    assert!((x[7] - 1.0).abs() <= 1e-5);
    assert!(x[8].abs() <= 1e-5);
    assert!(x[9].abs() <= 1e-5);
    assert!(x[10].abs() <= 1e-5);
    assert!((x[11] - 1.0).abs() <= 1e-5);
    assert!((x[12] - 1.0).abs() <= 1e-5);
    assert!((x[13] - 1.0).abs() <= 1e-5);
    assert!((x[14] - 1.0).abs() <= 1e-5);
}

// Fits a circle to generated points with a dense Jacobian.
#[test]
fn circle_fit_generated_points() {
    let center = (1.5, -0.5);
    let radius = 2.0;
    let points = 12;
    let mut samples = Vec::with_capacity(points);
    for i in 0..points {
        let theta = (i as f64) * (2.0 * std::f64::consts::PI / points as f64);
        samples.push((
            center.0 + radius * theta.cos(),
            center.1 + radius * theta.sin(),
        ));
    }

    let entries = dense_pattern_entries_1b(points, 3);
    let pattern = pattern_from_triplets_1b(points, 3, &entries);
    let (x, status) = solve_problem(
        pattern,
        vec![0.0, 0.0, 1.0],
        |x, out| {
            for (i, (px, py)) in samples.iter().enumerate() {
                let dx = px - x[0];
                let dy = py - x[1];
                out[i] = dx * dx + dy * dy - x[2] * x[2];
            }
        },
        |x, jac| {
            for col in 0..jac.ncols() {
                let rows = jac.row_indices_of_col(col).to_vec();
                let vals = jac.values_of_col_mut(col);
                for (idx, &row) in rows.iter().enumerate() {
                    let (px, py) = samples[row];
                    let val = match col {
                        0 => -2.0 * (px - x[0]),
                        1 => -2.0 * (py - x[1]),
                        _ => -2.0 * x[2],
                    };
                    vals[idx] = val;
                }
            }
        },
    );
    assert_converged(status);
    assert!((x[0] - center.0).abs() <= 1e-5);
    assert!((x[1] - center.1).abs() <= 1e-5);
    assert!((x[2] - radius).abs() <= 1e-5);
}

// Solves a banded linear chain with anchor and difference constraints.
#[test]
fn chain_constraints_linear() {
    let n = 20;
    let m = 2 * n - 1;
    let mut target = vec![0.0; n];
    for i in 0..n {
        target[i] = (i as f64 * 0.1).sin();
    }
    let entries = chain_pattern_entries_1b(n);
    let pattern = pattern_from_triplets_1b(m, n, &entries);
    let (x, status) = solve_problem(
        pattern,
        vec![0.0; n],
        |x, out| {
            for i in 0..n {
                out[i] = x[i] - target[i];
            }
            for i in 0..n - 1 {
                out[n + i] = (x[i + 1] - x[i]) - (target[i + 1] - target[i]);
            }
        },
        |_x, jac| {
            for col in 0..jac.ncols() {
                let rows = jac.row_indices_of_col(col).to_vec();
                let vals = jac.values_of_col_mut(col);
                for (idx, &row) in rows.iter().enumerate() {
                    let coeff = if row == col {
                        1.0
                    } else if row == n + col {
                        -1.0
                    } else if row + 1 == n + col {
                        1.0
                    } else {
                        0.0
                    };
                    vals[idx] = coeff;
                }
            }
        },
    );
    assert_converged(status);
    for i in 0..n {
        assert!((x[i] - target[i]).abs() <= 1e-5);
    }
}

// Recovers a deterministic sparse linear system from synthetic data.
#[test]
fn sparse_linear_system_recovery() {
    let (pattern, mut problem, x_star) = make_sparse_linear_problem(12, 5, 4, 0x5eed_u64);
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut x = vec![0.0; x_star.len()];
    let stats = solver
        .solve(&mut problem, &mut x, &solver_options(), None)
        .unwrap();
    assert_converged(stats.status);
    let mut max_err = 0.0_f64;
    for (xi, xref) in x.iter().zip(&x_star) {
        max_err = max_err.max((xi - xref).abs());
    }
    assert!(max_err <= 1e-5, "max error too high: {max_err}");
}

// Checks that a second solve stays within a small allocation budget.
#[test]
fn allocations() {
    let pattern = pattern_from_triplets_1b(1, 1, &[(1, 1)]);
    let mut solver = LmSolver::new(pattern.clone(), Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] - 1.0;
        },
        jac: |_x: &[f64], jac: &mut JacobianValuesMut<'_>| {
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
        },
    };
    let mut x = vec![0.0];
    solver
        .solve(&mut problem, &mut x, &solver_options(), None)
        .unwrap();

    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] - 1.0;
        },
        jac: |_x: &[f64], jac: &mut JacobianValuesMut<'_>| {
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 1.0);
        },
    };
    let mut x = vec![0.0];
    reset_alloc_counter();
    solver
        .solve(&mut problem, &mut x, &solver_options(), None)
        .unwrap();
    let alloc = allocated_bytes();
    assert!(alloc <= 50_000, "allocations too high: {alloc}");
}
