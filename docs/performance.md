# Measuring Series S performance

Build optimized code before comparing frame rates:

```powershell
cargo build --release -p victory
.\scripts\run-series-s.ps1 -GameExe 'D:\game\game.exe'
```

The launcher uses ordinary user input, six copy and six pipeline workers on
a 12-thread CPU, normal pipeline worker priority, batched card rehashes and
contiguous copies, GPU results fetched within their producing submissions,
and parallel copy-back merges. Overlapping copies and aliases of the same
physical section remain ordered. It restores the caller's
environment on exit. The default dump is next to the game under
`victory-performance/frametime.csv`; every run replaces that file. Override
it with `-FrametimeDump`. Save separate files for comparisons.

An optional 512 KiB window grid (`-WindowKiB 512`) with window tails reduced private memory by about
2.3 GiB on a 16 GiB PC. It can increase overlapping copies: use
the default 4096 KiB grid for the original behavior. `-CopyWorkers` and
`-PipelineWorkers` independently select worker counts. More workers alone
do not guarantee faster frames.

Opaque mesh fallbacks compact live triangles on the GPU before drawing.
Padded and culled entries do not invoke vertex or geometry shaders. The
mapping and indirect count stay on the GPU in bounded scratch storage;
no CPU readback is added. Blended pipelines retain their original draw
order. Use `-NoMeshCompaction`, or `VICTORY_MESH_COMPACT=0` when launching
directly, to compare the original mesh fallback.

Unchanged descriptor reads use concurrent view lookups and published heap
entries rather than the heap's writer lock.

The release-compatible frametime dumper is adapted from
[skornuta/victory-fork](https://github.com/skornuta/victory-fork).
`VICTORY_FRAMETIME_DUMP=1` uses the configured cache directory;
any other nonempty value names the output CSV. It records one row per
summary interval, including mean/worst frame time, copy, texture and heap
costs, draw counts, process working set, and process CPU utilization across
all logical processors. Lines beginning with `#` describe the machine and
environment. The GPU percentage estimates queue activity; `gpu_ms` includes
waiting for GPU work. Worker costs overlap with submission and copy-back
costs; do not add the buckets together as a frame budget.

Compare equivalent parts of the cutscene after loading and pipeline
compilation settle. Menu FPS and rows with dropped draws are insufficient
to establish a rendering performance improvement. Keep captures and debug
layers off for sustained measurements. Rendering correctness still needs
separate visual checks.

Summarize a chosen scene window with Python (no extra packages):

```powershell
python scripts/analyze-frametime.py 'D:\game\victory-performance\frametime.csv' --start-frame 2000 --end-frame 2300
```

Choose those frame bounds from your run, after loading and shader compilation.
The script uses total frames divided by total elapsed frame time, rather than
averaging interval FPS or taking the reciprocal of a median. It includes stalls
and dropped draws within the selected window. Only whole CSV intervals inside
the bounds are selected. `--exclude-frame 2048` excludes an interval containing
an explicitly captured frame; repeat for other captures. Cost buckets overlap.
Scripted menu input (`VICTORY_KEYS`) is available in ordinary release builds
for reproducible benchmarks; the interactive launcher clears it.

`VICTORY_CAPTURE_WORLD` saves a continuing timeline, including two images
every eight world frames. Do not enable it during an FPS benchmark. For a
visual check, use a separate run or `VICTORY_CAPTURE_FRAMES` without world
capture, and exclude those captured intervals from timing comparisons.
