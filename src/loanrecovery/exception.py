# ============================================================
# LOAN RECOVERY PROJECT — CUSTOM EXCEPTION HANDLER
# ============================================================
"""
Custom exception handling for loan recovery project.

Provides detailed error information including:
- File name where error occurred
- Line number
- Error message
- Full traceback
"""

import sys
from typing import Optional


def get_error_message(
    error: Exception,
    error_detail: sys
) -> str:
    """
    Extract detailed error information.

    Args:
        error: The exception that was raised
        error_detail: sys module for traceback info

    Returns:
        Formatted error message string with
        file name, line number, and error details
    """
    _, _, exc_tb = error_detail.exc_info()

    # Get file name where error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Get line number
    line_number = exc_tb.tb_lineno

    # Format message
    error_message = (
        f"\n{'='*60}\n"
        f"ERROR DETAILS:\n"
        f"  File    : {file_name}\n"
        f"  Line    : {line_number}\n"
        f"  Message : {str(error)}\n"
        f"{'='*60}"
    )

    return error_message


class LoanRecoveryException(Exception):
    """
    Custom exception for Loan Recovery Project.

    Provides detailed error context including
    file name, line number, and error message.

    Usage:
        try:
            # some code
        except Exception as e:
            raise LoanRecoveryException(e, sys)
    """

    def __init__(
        self,
        error_message: Exception,
        error_detail: sys
    ) -> None:
        """
        Initialize custom exception.

        Args:
            error_message: The caught exception
            error_detail: sys module for traceback
        """
        super().__init__(error_message)

        self.error_message = get_error_message(
            error=error_message,
            error_detail=error_detail
        )

    def __str__(self) -> str:
        """Return formatted error message."""
        return self.error_message

    def __repr__(self) -> str:
        """Return class representation."""
        return f"LoanRecoveryException({self.error_message})"
