# L0xRE v0.1.6.4 chat template hotfix

The template shipped in v0.1.6.4 uses a moving window for tool-call reasoning. As new messages arrive, the window can rewrite previously rendered history and reduce KV cache reuse. The hotfix keeps up to 400 characters of reasoning for a single `read_file` call in either of the first two assistant tool steps of the supplied conversation. Later tool steps render without their reasoning.

## Install

1. Download [l0xre-agent-v7.jinja](../templates/l0xre-agent-v7.jinja) using the raw file download. Save it alongside your kit or in a separate directory.
2. Restart L0xRE with `--chat-template-file /absolute/path/to/l0xre-agent-v7.jinja` in the server arguments. Replace the existing template argument if your startup command sets one. If you use the v0.1.6.4 Linux kit's `launch.py`, append `-- --chat-template-file /absolute/path/to/l0xre-agent-v7.jinja` after its regular options. If your command already uses `--`, add the override after that separator. Keep the bundled template in place so the kit's file integrity check still passes.
3. Confirm the server started and that `GET /props` returns the new content in `chat_template`. To roll back, remove the override or restore the prior template argument, then restart.

The hotfix file has SHA-256 `004389dbbdeb82b6b6034e55de483a3dd2fd31713a97a6a90df8c1e6c1fad598`.

## Qualification

The production RTX 3060 test used an intermediate prefix-stable baseline for the behavioral comparisons. The hotfix matched that baseline on all 14 loop replays, with every case reaching a different action within four steps. Valid first-response fixes improved from 9/13 to 13/13 on a coding fixture and from 2/6 to 6/6 after a repeated file read. Both versions applied and successfully retested all 13 timeout repairs in a disposable project.

Across three successive continuation requests, the hotfix reused about 100%, 96%, and 98% of the previous prompt. The shipped windowed template reused about 101%, 44%, and 29% in the same sequence. The numerator includes cached tokens generated in the previous response, so the ratio can exceed 100%. Runtime qualification covered Linux on the RTX 3060.
