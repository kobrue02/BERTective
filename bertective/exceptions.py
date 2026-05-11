"""Custom exceptions for BERTective."""


class ParsedPathNotExistError(FileNotFoundError):
    """Raised when a user-supplied string looks like a file path but does not exist."""


class MissingTrainingData(RuntimeError):
    """Raised when pre-computed feature vectors required for training are absent."""


class InvalidLabelError(ValueError):
    """Raised when a label value is outside the accepted set for its field."""


class CorpusNotFoundError(FileNotFoundError):
    """Raised when the corpus AVRO file cannot be found."""


class ResourceNotFoundError(FileNotFoundError):
    """Raised when a required data resource (JSON, pickle, etc.) is missing."""
