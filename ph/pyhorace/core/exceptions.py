"""Standard Horace and Herbert exception classes."""


class HoraceError(RuntimeError, ValueError):
    """Base exception for Horace runtime and invalid argument errors."""

    def __init__(self, *args: object) -> None:
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            self.error_id = args[0]
            self.message = args[1]
            super().__init__(f"{args[0]}: {args[1]}")
        else:
            self.error_id = ""
            self.message = str(args[0]) if args else ""
            super().__init__(*args)


HerbertError = HoraceError
