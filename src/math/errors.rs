use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum LinAlgError<T> {
    NotSquare,
    EmptyMatrix,
    NotSymmetric {
        matrix_name: String,
        max_asymmetry: T,
    },
    Singular {
        matrix_name: String,
        index: usize,
        pivot_abs: T,
        threshold: T,
    },
    NearSingular {
        matrix_name: String,
        index: usize,
        pivot_abs: T,
        threshold: T,
    },
    NotSpd {
        matrix_name: String,
        index: usize,
        diag_candidate: T,
        threshold: T,
    },
    ZeroDiagonal {
        matrix_name: String,
        index: usize,
    },
    DimensionMismatch {
        op: String,
        lhs: (usize, usize),
        rhs: (usize, usize),
    },
    Nan,
    Inf,
}

impl<T> Display for LinAlgError<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinAlgError::NotSquare => write!(f, "matrix must be square"),
            LinAlgError::EmptyMatrix => write!(f, "matrix must not be empty"),
            LinAlgError::NotSymmetric {
                matrix_name,
                max_asymmetry,
            } => write!(
                f,
                "matrix '{matrix_name}' is not symmetric; max asymmetry is {max_asymmetry}"
            ),
            LinAlgError::Singular {
                matrix_name,
                index,
                pivot_abs,
                threshold,
            } => write!(
                f,
                "matrix '{matrix_name}' is singular at pivot {index}; |pivot|={pivot_abs}, threshold={threshold}"
            ),
            LinAlgError::NearSingular {
                matrix_name,
                index,
                pivot_abs,
                threshold,
            } => write!(
                f,
                "matrix '{matrix_name}' is near-singular at pivot {index}; |pivot|={pivot_abs}, threshold={threshold}"
            ),
            LinAlgError::NotSpd {
                matrix_name,
                index,
                diag_candidate,
                threshold,
            } => write!(
                f,
                "matrix '{matrix_name}' is not SPD at diagonal {index}; candidate={diag_candidate}, threshold={threshold}"
            ),
            LinAlgError::ZeroDiagonal { matrix_name, index } => write!(
                f,
                "matrix '{matrix_name}' has an invalid near-zero diagonal at index {index}"
            ),
            LinAlgError::DimensionMismatch { op, lhs, rhs } => write!(
                f,
                "dimension mismatch in {op}; lhs is {}x{}, rhs is {}x{}",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),
            LinAlgError::Nan => write!(f, "encountered NaN in linear algebra computation"),
            LinAlgError::Inf => write!(f, "encountered Inf in linear algebra computation"),
        }
    }
}
