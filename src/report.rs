use comfy_table::{presets, Cell, CellAlignment, ContentArrangement, Table};

/// Solver termination status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    /// Infinity norm of the gradient is below tolerance.
    ConvergedGradient,
    /// Step norm is below tolerance.
    ConvergedStep,
    /// Cost is below tolerance.
    ConvergedCost,
    /// Reached the iteration limit without converging.
    MaxIterations,
    /// NaN or Inf encountered in the objective.
    NumericalFailure,
}

/// Summary statistics from a solve.
#[derive(Debug, Clone)]
pub struct SolverStats {
    /// Termination status.
    pub status: SolveStatus,
    /// Number of completed iterations.
    pub iterations: usize,
    /// Final cost, 0.5 * ||r(x)||^2.
    pub cost: f64,
    /// Infinity norm of the gradient, ||J^T r||_inf.
    pub grad_inf: f64,
    /// Step norm, ||p||_2.
    pub step_norm: f64,
    /// Final damping parameter.
    pub lambda: f64,
}

/// Per-iteration diagnostics.
#[derive(Debug, Clone)]
pub struct IterationReport {
    /// Iteration index, starting at 0.
    pub iteration: usize,
    /// Current cost, 0.5 * ||r(x)||^2.
    pub cost: f64,
    /// Trial cost after the candidate step.
    pub trial_cost: f64,
    /// Ratio of actual to predicted decrease.
    pub rho: f64,
    /// Current damping parameter.
    pub lambda: f64,
    /// Step norm, ||p||_2.
    pub step_norm: f64,
    /// Infinity norm of the gradient.
    pub grad_inf: f64,
    /// Whether the trial step was accepted.
    pub accepted: bool,
}

pub(crate) fn emit_line(line: &str) {
    if log::log_enabled!(log::Level::Info) {
        log::info!("{line}");
    } else {
        println!("{line}");
    }
}

/// Receives iteration updates from the solver.
pub trait Reporter {
    /// Called after each trial step is evaluated.
    fn on_iteration(&mut self, report: &IterationReport);
    /// Called once after the solver exits.
    fn on_finish(&mut self) {}
}

/// Reporter that prints a UTF-8 table to stdout or the log.
pub struct StdoutReporter {
    rows: Vec<IterationReport>,
}

impl StdoutReporter {
    /// Create a new stdout reporter.
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }
}

impl Default for StdoutReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Reporter for StdoutReporter {
    fn on_iteration(&mut self, report: &IterationReport) {
        self.rows.push(report.clone());
    }

    fn on_finish(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        if !log::log_enabled!(log::Level::Info) {
            println!();
        }
        let mut table = Table::new();
        table.load_preset(presets::UTF8_FULL);
        table.set_content_arrangement(ContentArrangement::Dynamic);
        table.set_header(vec![
            Cell::new("iter").set_alignment(CellAlignment::Right),
            Cell::new("cost").set_alignment(CellAlignment::Right),
            Cell::new("trial").set_alignment(CellAlignment::Right),
            Cell::new("rho").set_alignment(CellAlignment::Right),
            Cell::new("lambda").set_alignment(CellAlignment::Right),
            Cell::new("step").set_alignment(CellAlignment::Right),
            Cell::new("grad").set_alignment(CellAlignment::Right),
            Cell::new("accepted"),
        ]);
        for row in &self.rows {
            table.add_row(vec![
                Cell::new(row.iteration).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.4e}", row.cost)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.4e}", row.trial_cost)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.3}", row.rho)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.1e}", row.lambda)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.1e}", row.step_norm)).set_alignment(CellAlignment::Right),
                Cell::new(format!("{:.1e}", row.grad_inf)).set_alignment(CellAlignment::Right),
                Cell::new(if row.accepted { "yes" } else { "no" }),
            ]);
        }

        for line in table.to_string().lines() {
            emit_line(line);
        }
        self.rows.clear();
    }
}
