use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use faer_core::Parallelism;
use s_nnls_rs::{JacobianPattern, JacobianValuesMut, LmSolver, Problem, SolverOptions};

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

fn solver_options(verbose: bool) -> SolverOptions {
    SolverOptions {
        max_iters: 200,
        verbose,
        ..SolverOptions::default()
    }
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

fn bench_basic_linear(c: &mut Criterion) {
    let pattern = pattern_from_triplets_1b(1, 1, &[(1, 1)]);
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
    let x0 = vec![0.0];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("basic_linear", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_basic_nonlinear(c: &mut Criterion) {
    let pattern = pattern_from_triplets_1b(1, 1, &[(1, 1)]);
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] * x[0] * x[0] - 1.0;
        },
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
            let x1 = x[0];
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 3.0 * x1 * x1);
        },
    };
    let x0 = vec![0.5];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("basic_nonlinear", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_multidimensional_quadratics(c: &mut Criterion) {
    let pattern = pattern_from_triplets_1b(3, 2, &[(1, 1), (3, 1), (2, 2), (3, 2)]);
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
            let x1 = x[0];
            let x2 = x[1];
            out[0] = x1 * x1 - 1.0;
            out[1] = x2 * x2 - 4.0;
            out[2] = x1 * x2 - 2.0;
        },
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
            let x1 = x[0];
            let x2 = x[1];
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, 2.0 * x1);
            set_entry_1b(jac, 3, 1, x2);
            set_entry_1b(jac, 2, 2, 2.0 * x2);
            set_entry_1b(jac, 3, 2, x1);
        },
    };
    let x0 = vec![0.8, 1.7];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("multidimensional_quadratics", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_rosenbrock_residuals(c: &mut Criterion) {
    let pattern = pattern_from_triplets_1b(2, 2, &[(1, 1), (2, 1), (2, 2)]);
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
            let x1 = x[0];
            let x2 = x[1];
            out[0] = 1.0 - x1;
            out[1] = 10.0 * (x2 - x1 * x1);
        },
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
            let x1 = x[0];
            zero_jacobian(jac);
            set_entry_1b(jac, 1, 1, -1.0);
            set_entry_1b(jac, 2, 1, -20.0 * x1);
            set_entry_1b(jac, 2, 2, 10.0);
        },
    };
    let x0 = vec![-1.2, 1.0];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("rosenbrock_residuals", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_cad_axis_distance(c: &mut Criterion) {
    let pattern = pattern_from_triplets_1b(
        4,
        4,
        &[(1, 1), (4, 1), (2, 2), (4, 2), (4, 3), (3, 4), (4, 4)],
    );
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
            let x1 = x[0];
            let y1 = x[1];
            let x2 = x[2];
            let y2 = x[3];
            out[0] = x1 - 1.0;
            out[1] = y1 - 1.0;
            out[2] = y2;
            out[3] = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) - 4.0;
        },
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
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
    };
    let x0 = vec![0.8, 1.2, 2.6, 0.1];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("cad_axis_distance", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_cad_parallel_lines(c: &mut Criterion) {
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
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
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
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
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
    };
    let x0 = vec![0.1, -0.1, 2.1, 0.2, -0.1, 1.1, 1.9, 1.2];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("cad_parallel_lines", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_cad_perpendicular_lines(c: &mut Criterion) {
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
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
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
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
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
    };
    let x0 = vec![0.1, -0.1, 2.1, 0.1, 1.1, 0.1, 0.9, 2.1];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("cad_perpendicular_lines", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_cad_tangent_circle(c: &mut Criterion) {
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
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
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
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
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
    };
    let x0 = vec![0.1, -0.1, 2.1, 0.1, 0.9, 0.8, 0.9, 2.1, 0.9];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("cad_tangent_circle", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

fn bench_cad_complex_constraints(c: &mut Criterion) {
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
    let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
    let mut problem = FnProblem {
        res: |x: &[f64], out: &mut [f64]| {
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
        jac: |x: &[f64], jac: &mut JacobianValuesMut<'_>| {
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
    };
    let x0 = vec![
        0.1, -0.1, 2.1, 0.2, -0.1, 1.1, 1.9, 0.9, 0.2, 0.1, -0.1, 1.2, 0.9, 1.2, 1.1,
    ];
    let mut x = x0.clone();
    let opts_verbose = solver_options(true);
    let opts_quiet = solver_options(false);
    let mut first = true;
    c.bench_function("cad_complex_constraints", |b| {
        b.iter(|| {
            x.copy_from_slice(&x0);
            let opts = if first {
                first = false;
                &opts_verbose
            } else {
                &opts_quiet
            };
            solver.solve(&mut problem, &mut x, opts, None).unwrap();
            black_box(&x);
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_millis(1000));
    targets =
        bench_basic_linear,
        bench_basic_nonlinear,
        bench_multidimensional_quadratics,
        bench_rosenbrock_residuals,
        bench_cad_axis_distance,
        bench_cad_parallel_lines,
        bench_cad_perpendicular_lines,
        bench_cad_tangent_circle,
        bench_cad_complex_constraints
}
criterion_main!(benches);
