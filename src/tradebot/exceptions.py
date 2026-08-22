from __future__ import annotations


class TradebotError(RuntimeError): pass
class DataIntegrityError(TradebotError): pass
class SchemaViolationError(TradebotError):
    def __init__(self, message: str, schema_name: str = '') -> None:
        self.schema_name = schema_name
        super().__init__(f'[{schema_name}] {message}' if schema_name else message)
class LookaheadError(TradebotError): pass
class PublicationLagViolation(LookaheadError):
    def __init__(self, series, bar_time, pub_time):
        super().__init__(f'Pub-lag: {series!r} at {bar_time} published at {pub_time}')
class FeatureSchemaMismatch(TradebotError):
    def __init__(self, symbol, missing=None, extra=None):
        self.symbol = symbol; self.missing = missing or []; self.extra = extra or []
        super().__init__(f'[{symbol}] Schema mismatch missing={self.missing[:5]} extra={self.extra[:5]}')
class StageInputMissing(TradebotError):
    def __init__(self, stage, path): super().__init__(f'Stage {stage!r} missing: {path}')
class DeterminismError(TradebotError): pass
class DrawdownBreakerActive(TradebotError): pass
class PositionLimitExceeded(TradebotError): pass
class ConfigurationError(TradebotError): pass
class ModelNotFound(TradebotError): pass
class LineageHashMismatch(TradebotError): pass
