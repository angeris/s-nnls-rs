use core::fmt;

use dyn_stack::{GlobalPodBuffer, PodStack, ReborrowMut};
use std::time::{Duration, Instant};
use faer_core::mat;
use faer_core::{Conj, Parallelism};
use faer_core::sparse::SparseColMatRef;
use faer_sparse::qr::{factorize_symbolic_qr, QrSymbolicParams, SymbolicQr};

use crate::pattern::{AugmentedPattern, JacobianPattern, PatternError};
use crate::report::{emit_line, IterationReport, Reporter, SolveStatus, SolverStats, StdoutReporter};

/// Errors while constructing or running the solver.
#[derive(Debug)]
pub enum SolverError {
    /// The sparsity pattern is invalid.
    Pattern(PatternError),
    /// The pattern has zero rows or columns.
    InvalidDimensions { nrows: usize, ncols: usize },
    /// faer reported an error during factorization or solve.
    Faer(faer_sparse::FaerError),
    /// Workspace requirement overflowed.
    WorkspaceOverflow,
    /// Workspace allocation failed.
    WorkspaceAlloc,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pattern(err) => write!(f, "invalid sparsity pattern: {err}"),
            Self::InvalidDimensions { nrows, ncols } => {
                write!(f, "invalid dimensions: nrows={nrows}, ncols={ncols}")
            }
            Self::Faer(err) => write!(f, "faer error: {err:?}"),
            Self::WorkspaceOverflow => write!(f, "workspace size overflow"),
            Self::WorkspaceAlloc => write!(f, "workspace allocation failed"),
        }
    }
}

impl std::error::Error for SolverError {}

/// Errors specific to a solve call.
#[derive(Debug)]
pub enum SolveError {
    /// The provided x has the wrong length.
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "x length {actual} does not match expected {expected}")
            }
        }
    }
}

impl std::error::Error for SolveError {}

/// Options controlling the Levenberg-Marquardt solve.
#[derive(Debug, Clone)]
pub struct SolverOptions {
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Converge when ||J^T r||_inf <= grad_tol.
    pub grad_tol: f64,
    /// Converge when ||p||_2 <= step_tol * (||x||_2 + step_tol).
    pub step_tol: f64,
    /// Converge when 0.5 * ||r||^2 <= cost_tol.
    pub cost_tol: f64,
    /// Initial damping parameter.
    pub lambda_init: f64,
    /// Minimum damping parameter.
    pub lambda_min: f64,
    /// Maximum damping parameter.
    pub lambda_max: f64,
    /// Emit per-iteration diagnostics to stdout by default.
    pub verbose: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iters: 50,
            grad_tol: 1e-8,
            step_tol: 1e-10,
            cost_tol: 1e-12,
            lambda_init: 1e-3,
            lambda_min: 1e-12,
            lambda_max: 1e12,
            verbose: false,
        }
    }
}

/// Nonlinear least squares problem with residuals r(x) and Jacobian J(x).
pub trait Problem {
    /// Fill residuals r(x).
    fn residuals(&mut self, x: &[f64], residuals: &mut [f64]);
    /// Fill Jacobian values in column order for the given sparsity pattern.
    fn jacobian(&mut self, x: &[f64], jacobian: &mut JacobianValuesMut<'_>);

    /// Optional combined evaluation; default calls residuals then jacobian.
    fn residuals_and_jacobian(
        &mut self,
        x: &[f64],
        residuals: &mut [f64],
        jacobian: &mut JacobianValuesMut<'_>,
    ) {
        self.residuals(x, residuals);
        self.jacobian(x, jacobian);
    }
}

/// Mutable view of Jacobian values matching the sparsity pattern.
pub struct JacobianValuesMut<'a> {
    values: &'a mut [f64],
    col_ptrs: &'a [usize],
    row_indices: &'a [usize],
    diag_positions: &'a [usize],
    nrows: usize,
}

impl<'a> JacobianValuesMut<'a> {
    pub(crate) fn new(values: &'a mut [f64], pattern: &'a AugmentedPattern) -> Self {
        Self {
            values,
            col_ptrs: pattern.col_ptrs(),
            row_indices: pattern.row_indices(),
            diag_positions: pattern.diag_positions(),
            nrows: pattern.jacobian_rows(),
        }
    }

    /// Number of residuals (rows in J).
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of parameters (columns in J).
    pub fn ncols(&self) -> usize {
        self.diag_positions.len()
    }

    /// Sorted row indices for the given column.
    pub fn row_indices_of_col(&self, col: usize) -> &[usize] {
        let start = self.col_ptrs[col];
        let end = self.diag_positions[col];
        &self.row_indices[start..end]
    }

    /// Mutable values for the given column, aligned with row_indices_of_col.
    pub fn values_of_col_mut(&mut self, col: usize) -> &mut [f64] {
        let start = self.col_ptrs[col];
        let end = self.diag_positions[col];
        &mut self.values[start..end]
    }
}

/// Sparse Levenberg-Marquardt solver for min 0.5 * ||r(x)||^2.
///
/// The Jacobian sparsity pattern is fixed at construction and must match the
/// values provided by `Problem::jacobian`.
pub struct LmSolver {
    pattern: JacobianPattern,
    augmented: AugmentedPattern,
    symbolic_qr: SymbolicQr<usize>,
    qr_indices: Vec<usize>,
    qr_values: Vec<f64>,
    qr_stack: GlobalPodBuffer,
    parallelism: Parallelism,
    values: Vec<f64>,
    residuals: Vec<f64>,
    trial_residuals: Vec<f64>,
    rhs: Vec<f64>,
    gradient: Vec<f64>,
    x_trial: Vec<f64>,
}

enum ReporterSlot<'a> {
    External(&'a mut dyn Reporter),
    Local(StdoutReporter),
    None,
}

impl<'a> ReporterSlot<'a> {
    fn new(reporter: Option<&'a mut dyn Reporter>, verbose: bool) -> Self {
        match reporter {
            Some(r) => Self::External(r),
            None if verbose => Self::Local(StdoutReporter::new()),
            None => Self::None,
        }
    }

    fn as_mut(&mut self) -> Option<&mut dyn Reporter> {
        match self {
            Self::External(r) => Some(*r),
            Self::Local(r) => Some(r),
            Self::None => None,
        }
    }
}

impl LmSolver {
    /// Create a solver for the given sparsity pattern and parallelism mode.
    pub fn new(
        pattern: JacobianPattern,
        parallelism: Parallelism,
    ) -> Result<Self, SolverError> {
        if pattern.nrows() == 0 || pattern.ncols() == 0 {
            return Err(SolverError::InvalidDimensions {
                nrows: pattern.nrows(),
                ncols: pattern.ncols(),
            });
        }

        let augmented = AugmentedPattern::new(&pattern);
        let symbolic_qr = factorize_symbolic_qr(
            augmented.as_symbolic(),
            QrSymbolicParams::default(),
        )
        .map_err(SolverError::Faer)?;

        let factor_req = symbolic_qr
            .factorize_numeric_qr_req::<f64>(parallelism)
            .map_err(|_| SolverError::WorkspaceOverflow)?;
        let solve_req = symbolic_qr
            .solve_in_place_req::<f64>(1, parallelism)
            .map_err(|_| SolverError::WorkspaceOverflow)?;
        let req = factor_req
            .try_or(solve_req)
            .map_err(|_| SolverError::WorkspaceOverflow)?;
        let qr_stack = GlobalPodBuffer::try_new(req).map_err(|_| SolverError::WorkspaceAlloc)?;

        let values = vec![0.0; augmented.row_indices().len()];
        let qr_indices = vec![0usize; symbolic_qr.len_indices()];
        let qr_values = vec![0.0; symbolic_qr.len_values()];

        let n = pattern.ncols();
        let m = pattern.nrows();
        Ok(Self {
            pattern,
            augmented,
            symbolic_qr,
            qr_indices,
            qr_values,
            qr_stack,
            parallelism,
            values,
            residuals: vec![0.0; m],
            trial_residuals: vec![0.0; m],
            rhs: vec![0.0; m + n],
            gradient: vec![0.0; n],
            x_trial: vec![0.0; n],
        })
    }

    /// Return the sparsity pattern used by this solver.
    pub fn jacobian_pattern(&self) -> &JacobianPattern {
        &self.pattern
    }

    /// Solve for x in-place using Levenberg-Marquardt.
    ///
    /// Minimizes 0.5 * ||r(x)||^2 and returns summary statistics.
    pub fn solve(
        &mut self,
        problem: &mut impl Problem,
        x: &mut [f64],
        options: &SolverOptions,
        reporter: Option<&mut dyn Reporter>,
    ) -> Result<SolverStats, SolveError> {
        let n = self.pattern.ncols();
        if x.len() != n {
            return Err(SolveError::DimensionMismatch {
                expected: n,
                actual: x.len(),
            });
        }
        let start_time = options.verbose.then(Instant::now);
        let mut reporter = ReporterSlot::new(reporter, options.verbose);

        let m = self.pattern.nrows();
        let mut lambda = clamp_lambda(options.lambda_init, options);

        // Evaluate r(x) and J(x) at the current iterate.
        {
            let mut jac = JacobianValuesMut::new(&mut self.values, &self.augmented);
            problem.residuals_and_jacobian(x, &mut self.residuals, &mut jac);
        }

        // Cost is 0.5 * ||r||^2.
        let mut cost = 0.5 * dot(&self.residuals, &self.residuals);
        if !cost.is_finite() {
            let stats = SolverStats {
                status: SolveStatus::NumericalFailure,
                iterations: 0,
                cost,
                grad_inf: f64::INFINITY,
                step_norm: f64::INFINITY,
                lambda,
            };
            return Ok(finish_stats(stats, x, start_time, &mut reporter));
        }

        let mut last_step_norm = 0.0;
        let mut last_grad_inf = 0.0;

        for iter in 0..options.max_iters {
            // Gradient g = J^T r; check convergence.
            compute_gradient(
                &mut self.gradient,
                &self.augmented,
                &self.values,
                &self.residuals,
            );
            let grad_inf = max_abs(&self.gradient);
            last_grad_inf = grad_inf;
            if grad_inf <= options.grad_tol {
                let stats = SolverStats {
                    status: SolveStatus::ConvergedGradient,
                    iterations: iter,
                    cost,
                    grad_inf,
                    step_norm: last_step_norm,
                    lambda,
                };
                return Ok(finish_stats(stats, x, start_time, &mut reporter));
            }
            if cost <= options.cost_tol {
                let stats = SolverStats {
                    status: SolveStatus::ConvergedCost,
                    iterations: iter,
                    cost,
                    grad_inf,
                    step_norm: last_step_norm,
                    lambda,
                };
                return Ok(finish_stats(stats, x, start_time, &mut reporter));
            }

            // Solve LM system via augmented QR: [J; sqrt(lambda) I] p = [-r; 0].
            let diag = lambda.sqrt();
            for &pos in self.augmented.diag_positions() {
                self.values[pos] = diag;
            }

            let a = SparseColMatRef::<'_, usize, f64>::new(
                self.augmented.as_symbolic(),
                self.values.as_slice(),
            );
            for i in 0..m {
                self.rhs[i] = -self.residuals[i];
            }
            for i in m..m + n {
                self.rhs[i] = 0.0;
            }

            let mut stack = PodStack::new(&mut self.qr_stack);
            let qr = self.symbolic_qr.factorize_numeric_qr::<f64>(
                &mut self.qr_indices,
                &mut self.qr_values,
                a,
                self.parallelism,
                stack.rb_mut(),
            );

            let mut rhs_mat = mat::from_column_major_slice_mut::<f64>(&mut self.rhs, m + n, 1);
            let mut stack = PodStack::new(&mut self.qr_stack);
            qr.solve_in_place_with_conj(
                Conj::No,
                rhs_mat.rb_mut(),
                self.parallelism,
                stack.rb_mut(),
            );

            // Step p is the first n entries of the solved system.
            let step = &self.rhs[..n];
            let step_norm = l2_norm(step);
            last_step_norm = step_norm;
            let x_norm = l2_norm(x);
            if step_norm <= options.step_tol * (x_norm + options.step_tol) {
                let stats = SolverStats {
                    status: SolveStatus::ConvergedStep,
                    iterations: iter,
                    cost,
                    grad_inf,
                    step_norm,
                    lambda,
                };
                return Ok(finish_stats(stats, x, start_time, &mut reporter));
            }

            // Predicted vs actual decrease gives rho for acceptance.
            let predicted = -0.5 * dot(step, &self.gradient);
            let mut trial_cost = cost;
            let mut rho = 0.0;
            let mut accepted = false;

            if predicted > 0.0 {
                for i in 0..n {
                    self.x_trial[i] = x[i] + step[i];
                }
                problem.residuals(&self.x_trial, &mut self.trial_residuals);
                trial_cost = 0.5 * dot(&self.trial_residuals, &self.trial_residuals);

                let actual = cost - trial_cost;
                if actual.is_finite() {
                    rho = actual / predicted;
                    accepted = rho > 0.0 && trial_cost.is_finite();
                }
            }

            if let Some(reporter) = reporter.as_mut() {
                reporter.on_iteration(&IterationReport {
                    iteration: iter,
                    cost,
                    trial_cost,
                    rho,
                    lambda,
                    step_norm,
                    grad_inf,
                    accepted,
                });
            }

            // Accept step and refresh J, or increase damping.
            if accepted {
                x.copy_from_slice(&self.x_trial);
                self.residuals.copy_from_slice(&self.trial_residuals);
                cost = trial_cost;
                lambda = clamp_lambda(update_lambda(lambda, rho), options);

                let mut jac = JacobianValuesMut::new(&mut self.values, &self.augmented);
                problem.jacobian(x, &mut jac);
            } else {
                lambda = clamp_lambda(lambda * 2.0, options);
            }
        }

        let stats = SolverStats {
            status: SolveStatus::MaxIterations,
            iterations: options.max_iters,
            cost,
            grad_inf: last_grad_inf,
            step_norm: last_step_norm,
            lambda,
        };
        Ok(finish_stats(stats, x, start_time, &mut reporter))
    }
}

fn clamp_lambda(lambda: f64, options: &SolverOptions) -> f64 {
    lambda
        .max(options.lambda_min)
        .min(options.lambda_max)
        .max(f64::MIN_POSITIVE)
}

fn update_lambda(lambda: f64, rho: f64) -> f64 {
    if rho > 0.0 {
        let t = 1.0 - (2.0 * rho - 1.0).powi(3);
        lambda * t.max(1.0 / 3.0)
    } else {
        lambda * 2.0
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

fn l2_norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn max_abs(x: &[f64]) -> f64 {
    let mut max = 0.0;
    for &v in x {
        let v = v.abs();
        if v > max {
            max = v;
        }
    }
    max
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs >= 1.0 {
        format!("{:.3} s", secs)
    } else if secs >= 1e-3 {
        format!("{:.3} ms", secs * 1e3)
    } else if secs >= 1e-6 {
        format!("{:.3} us", secs * 1e6)
    } else {
        format!("{:.0} ns", secs * 1e9)
    }
}

fn finish_stats(
    stats: SolverStats,
    x: &[f64],
    start_time: Option<Instant>,
    reporter: &mut ReporterSlot<'_>,
) -> SolverStats {
    if let Some(reporter) = reporter.as_mut() {
        reporter.on_finish();
    }
    if let Some(start) = start_time {
        let elapsed = format_duration(start.elapsed());
        emit_line(&format!("time: {elapsed}"));
        let _ = x;
    }
    stats
}

fn compute_gradient(
    grad: &mut [f64],
    pattern: &AugmentedPattern,
    values: &[f64],
    residuals: &[f64],
) {
    grad.fill(0.0);
    let ncols = pattern.ncols();
    for col in 0..ncols {
        let start = pattern.col_ptrs()[col];
        let end = pattern.diag_positions()[col];
        let mut sum = 0.0;
        for idx in start..end {
            let row = pattern.row_indices()[idx];
            sum += values[idx] * residuals[row];
        }
        grad[col] = sum;
    }
}

impl From<PatternError> for SolverError {
    fn from(err: PatternError) -> Self {
        Self::Pattern(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer_core::Parallelism;

    struct OneDProblem;

    impl Problem for OneDProblem {
        fn residuals(&mut self, x: &[f64], residuals: &mut [f64]) {
            residuals[0] = x[0] - 2.0;
        }

        fn jacobian(&mut self, _x: &[f64], jacobian: &mut JacobianValuesMut<'_>) {
            jacobian.values_of_col_mut(0)[0] = 1.0;
        }
    }

    #[test]
    fn solves_simple_problem() {
        let pattern = JacobianPattern::new(1, 1, vec![0, 1], vec![0]).unwrap();
        let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
        let mut x = [0.0];
        let mut problem = OneDProblem;
        let stats = solver.solve(&mut problem, &mut x, &SolverOptions::default(), None);
        let stats = stats.unwrap();
        assert!(matches!(
            stats.status,
            SolveStatus::ConvergedGradient
                | SolveStatus::ConvergedStep
                | SolveStatus::ConvergedCost
        ));
        assert!((x[0] - 2.0).abs() < 1e-6);
    }
}
