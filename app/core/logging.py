"""Structured logging setup."""
import logging, sys, json, os

class JsonFormatter(logging.Formatter):
    def format(self, record):  # pragma: no cover
        base = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "__dict__", {}).items():
            if k.startswith("_"):
                continue
        return json.dumps(base)

def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "json") == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
