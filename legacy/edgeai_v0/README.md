# EdgeAI v0 — Legacy

This folder contains the original EdgeAI framework prototype (v0)
built during the first phase of the EDI project.

The v0 prototype proved the core concept of device-agnostic edge inference
and directly informed the architecture of EdgeFlow v1.0.

Key learnings from v0 that shaped EdgeFlow:
- Auto device classification using psutil worked well
- Pre-bundling all models was not scalable (solved in v1 with modular install)
- Python-only runtime excluded most IoT devices (solved in v1 with C++ runtime)
- No CLI made it hard to use (solved in v1 with full CLI)

See the main EdgeFlow repo for the production framework.
