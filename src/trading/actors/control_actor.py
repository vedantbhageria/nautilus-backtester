"""ControlActor: receives commands from the dashboard (via the message bus /
external Redis streams) and drives the engine — subscribe/unsubscribe symbols,
request historical data, etc.

This is the command surface the dashboard talks to. Implementation TBD;
this file exists to give the component a home in the package layout.
"""
