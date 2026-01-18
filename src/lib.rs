//! Sparse nonlinear least squares solved with a Levenberg-Marquardt (LM) step.
//!
//! This crate minimizes `0.5 * ||r(x)||^2` for residuals `r(x)` with Jacobian `J(x)`.
//! It assumes a fixed sparsity pattern for `J` in compressed sparse column (CSC) form
//! and reuses allocations across solves.
//!
//! How it works (high level):
//! - Build the augmented system `[J; sqrt(lambda) I] p = [-r; 0]`.
//! - Solve it with sparse QR.
//! - Update `lambda` using the ratio of actual to predicted decrease.
//!
//! Calling it:
//! - Create a `JacobianPattern` (0-based indices, sorted rows per column).
//! - Implement `Problem` to fill residuals and Jacobian values.
//! - Call `LmSolver::solve` and inspect `SolverStats`.
//!
//! Example:
//! ```rust,no_run
//! use s_nnls_rs::{JacobianPattern, JacobianValuesMut, LmSolver, Problem, SolverOptions};
//! use faer_core::Parallelism;
//!
//! struct OneD;
//! impl Problem for OneD {
//!     fn residuals(&mut self, x: &[f64], r: &mut [f64]) {
//!         r[0] = x[0] - 1.0;
//!     }
//!     fn jacobian(&mut self, _x: &[f64], jac: &mut JacobianValuesMut<'_>) {
//!         jac.values_of_col_mut(0)[0] = 1.0;
//!     }
//! }
//!
//! let pattern = JacobianPattern::new(1, 1, vec![0, 1], vec![0]).unwrap();
//! let mut solver = LmSolver::new(pattern, Parallelism::None).unwrap();
//! let mut problem = OneD;
//! let mut x = vec![0.0];
//! let stats = solver.solve(&mut problem, &mut x, &SolverOptions::default(), None).unwrap();
//! assert!(stats.cost.is_finite());
//! ```

mod pattern;
mod report;
mod solver;

pub use pattern::{JacobianPattern, PatternError};
pub use report::{IterationReport, Reporter, SolveStatus, SolverStats, StdoutReporter};
pub use solver::{JacobianValuesMut, LmSolver, Problem, SolveError, SolverError, SolverOptions};
