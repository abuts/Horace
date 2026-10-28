"""Standard Horace and Herbert exception classes."""


class HoraceError(ValueError, RuntimeError):
    """Base exception for Horace runtime and invalid argument errors."""

    def __init__(self, *args: object) -> None:
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            self.error_id = args[0]
            self.identifier = args[0]
            self.message = args[1]
            full_msg = f"{args[0]}: {args[1]}" if args[1] else args[0]
            super().__init__(full_msg)
        elif len(args) == 1 and isinstance(args[0], str):
            self.error_id = args[0]
            self.identifier = args[0]
            self.message = args[0]
            super().__init__(args[0])
        else:
            self.error_id = ""
            self.identifier = ""
            self.message = str(args[0]) if args else ""
            super().__init__(*args)


HerbertError = HoraceError
