"""Opt-in, pre-trip geographic analysis. No runtime registration or persistence access."""


def safely_evaluate(operation, *args, **kwargs):
    """Optional caller boundary: enrichment failure is data, never a state mutation.

    No reminder integration is performed here. Callers retain their existing
    response and state when this independent result reports unavailable.
    """
    try:
        return {"status": "ok", "result": operation(*args, **kwargs)}
    except Exception as error:
        # Exception messages can contain provider credentials or local paths.
        return {"status": "unavailable", "error_type": type(error).__name__}
