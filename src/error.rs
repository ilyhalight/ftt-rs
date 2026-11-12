use std::fmt;

/// Errors returned while loading or using a model.
#[derive(Debug)]
pub enum Error {
    /// An I/O error, typically while opening or mapping the model file.
    Io(std::io::Error),
    /// The file is not a valid fastText model, or is truncated.
    Format(String),
    /// The model is valid but uses a feature this crate does not implement.
    Unsupported(String),
    /// An invalid argument was passed to the API.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(msg) => write!(f, "invalid fastText model: {msg}"),
            Error::Unsupported(msg) => write!(f, "unsupported model: {msg}"),
            Error::Invalid(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
