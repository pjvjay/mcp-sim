# mcp-sim

LLM-as-a-judge simulations for MCP servers. Give it a **role**, a **goal**, **instructions** and
an **expected outcome**; it plans several paths through the server's tools, runs an agent down
each of them, and has an independent judge decide whether the goal was reached honestly.

Design: [docs/DESIGN.md](docs/DESIGN.md). Implementation in progress.
