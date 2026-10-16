# Skala

Skala upscales images and videos 2x-8x from the command line. It trains nothing
itself: it orchestrates two external tools — [Real-ESRGAN
(ncnn/Vulkan)](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) for the
super-resolution pass, and ffmpeg/ffprobe for video demuxing and muxing. Runs on
Windows and Linux, Python 3.10+.

`skala setup` downloads the engine binary and its bundled `models/` folder into a
per-user data directory, so the only tool you must provide yourself is
ffmpeg/ffprobe — and only for video.

One command does the obvious thing:

```console
$ skala photo.png --scale 2
```

That writes `photo_skala_2x.png` next to the input. Point it at a video or a
folder and it figures out the rest.

## Requirements

| Requirement | Needed for | Notes |
| --- | --- | --- |
| Python 3.10+ | everything | |
| `realesrgan-ncnn-vulkan` | everything | the binary **and** its `models/` folder; `skala setup` installs it |
| `ffmpeg` + `ffprobe` | video only | images work without them |

Skala never bundles these: `skala setup` fetches the engine and its models on
demand. Run `skala doctor` to check what the current machine is missing, and see
[External tools](#external-tools) for where Skala looks.

## Install

From source (Skala is not on PyPI):

```console
$ git clone https://github.com/santstyle/Skala.git
$ cd Skala
$ python -m pip install -e .
```

For development (adds pytest, ruff, mypy):

```console
$ python -m pip install -e ".[dev]"
```

## Quick start

```console
# Download the engine + its models into the Skala data directory
$ skala setup

# Upscale an image 4x (default scale) using the default photo model
$ skala photo.png

# 2x with an explicit output path
$ skala photo.png --scale 2 --output big.png

# Upscale every image in a folder into another folder
$ skala ./shots --scale 4 --output ./upscaled

# Upscale a video, dropping the audio track
$ skala clip.mp4 --scale 2 --audio drop

# See what would happen without running the engine
$ skala clip.mp4 --scale 8 --dry-run

# Let Skala ask for the file and the scale
$ skala

# Check that the external tools are reachable
$ skala doctor
```

`skala run` is the default command, so `skala photo.png` and
`skala run photo.png` are the same thing. Run `skala` with no arguments in a
terminal and it [asks](#interactive-mode) for the file and the scale.

## Commands

| Command | Purpose |
| --- | --- |
| `skala [SOURCE] [options]` | Upscale a file or folder (default command) |
| `skala run [SOURCE] [options]` | Same, spelled out explicitly |
| `skala setup [options]` | Download the engine binary and its `models/` folder |
| `skala doctor` | Report whether the external tools are available |

`skala --version` prints the version and exits.

## Installing the engine

`skala setup` downloads the newest Real-ESRGAN engine release for your OS and
unpacks it — binary plus `models/` — into the Skala data directory:

```console
# Show the plan without downloading anything
$ skala setup --dry-run

# Install or upgrade the engine
$ skala setup

# Reinstall even if an engine is already present
$ skala setup --force

# Pin a specific release tag
$ skala setup --version v0.2.5.0
```

| Option | Meaning |
| --- | --- |
| `--version TAG` | Pin a specific engine release tag. |
| `--force` | Reinstall even if an engine is already present. |
| `--dry-run` | Print the plan and exit; no network access, no writes. |
| `--home PATH` | Install root. Defaults to the Skala data directory. |

The data directory is `%LOCALAPPDATA%\skala` on Windows and
`$XDG_DATA_HOME/skala` (usually `~/.local/share/skala`) elsewhere; override it
with `SKALA_HOME` or `--home`. The binary lands in `<data dir>/engine`, with its
`models/` folder beside it, where the rest of Skala picks it up automatically.

## Interactive mode

Run `skala` with no arguments in a terminal and Skala walks you through the job
instead of requiring flags:

```console
$ skala
Skala interactive mode
Answer a few prompts; press Ctrl+C to cancel.

File or folder to upscale: ./shots
Upscale factor (2, 3, 4 or 8) [4]: 2
```

It validates the path and the scale, re-prompting until each is acceptable, then
runs exactly as if you had typed `skala ./shots --scale 2` — the same `run` code
path, so `skala.toml` and the other options still apply.

Interactive mode needs a real terminal. When stdin is not a TTY (piped input, CI,
a task runner) a bare `skala` prints the help text instead of blocking on a
prompt that will never be answered.

## Options

| Option | Type | Meaning |
| --- | --- | --- |
| `--scale`, `-s` | integer | Upscale factor: `2`, `3`, `4` or `8`. Default `4`. |
| `--type` | text | `auto` (default), `image` or `video`. Overrides suffix detection. |
| `--model`, `-m` | text | Engine model name. Default depends on the target kind. |
| `--output`, `-o` | path | Output file, or output **directory** for a folder input. |
| `--format`, `-f` | text | Image output format: `png`, `jpg` or `webp`. |
| `--tile` | integer | Engine tile size; `0` means auto. Larger tiles need more VRAM. |
| `--model-dir` | path | Folder holding the engine's `models/` directory. |
| `--gpu-id`, `-g` | integer | GPU device index. |
| `--threads`, `-j` | integer | Engine thread count (>= 1). |
| `--crf` | integer | Video quality, 0-51, lower is better. Default `18`. |
| `--video-codec` | text | ffmpeg video encoder. Default `libx264`. |
| `--audio` | text | `copy` (default), `drop` or `aac`. |
| `--engine-path` | path | Explicit engine binary. |
| `--ffmpeg-path` | path | Explicit ffmpeg executable. |
| `--ffprobe-path` | path | Explicit ffprobe executable. |
| `--workdir` | path | Parent directory for scratch files. |
| `--overwrite` / `--no-overwrite` | flag | Replace existing output files. |
| `--keep-temp` / `--no-keep-temp` | flag | Keep scratch files after a run. |
| `--verbose`, `-v` / `--no-verbose` | flag | Debug logging. |
| `--dry-run` | flag | Print the plan, then exit without touching anything. |
| `--config` | path | Explicit `skala.toml` path. |

Options must come **after** the input path (`skala photo.png --scale 2`), because
the input is a positional argument.

## Scales and models

Skala accepts `2`, `3`, `4` and `8`. Each shipped model only supports some
scales natively, so Skala chains engine passes and, when needed, finishes with a
LANCZOS resize:

| Requested | Engine passes |
| --- | --- |
| 2x | native `2x` pass if the model has one, otherwise one `4x` pass downscaled to 2x |
| 3x | native `3x` pass if the model has one, otherwise one `4x` pass downscaled to 3x |
| 4x | one `4x` pass |
| 8x | `4x` then `2x` when the model has a native 2x pass, otherwise one `4x` pass followed by a LANCZOS 2x upscale |

8x always emits a warning: it is memory-hungry and, on models without a native
2x pass, the final 2x step is a plain interpolation rather than a network pass.

| Model | Native scales | Kind | Default for |
| --- | --- | --- | --- |
| `realesrgan-x4plus` | 4 | photo | images |
| `realesrgan-x4plus-anime` | 4 | anime | |
| `realesrnet-x4plus` | 4 | photo | |
| `realesr-animevideov3` | 2, 3, 4 | video | videos |

Use `--model` to pick a different one.

## Configuration file

Drop a `skala.toml` in the directory you run from and Skala picks it up
automatically. `git diff`-able, no hidden global state. Use `--config PATH` to
point at a specific file.

```toml
# skala.toml
model = "realesrgan-x4plus-anime"
tile = 200
gpu_id = 0
threads = 4
crf = 20
video_codec = "libx264"
audio = "copy"
engine_path = "/opt/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan"
workdir = "/tmp/skala-work"
keep_temp = false
overwrite = true
verbose = false
```

Recognised keys: `model`, `engine_path`, `ffmpeg_path`, `ffprobe_path`, `tile`,
`gpu_id`, `threads`, `workdir`, `keep_temp`, `overwrite`, `verbose`, `crf`,
`video_codec`, `audio`. Unknown keys are an error. `scale` is **not** a config
key — always pass it on the command line.

Precedence is: built-in defaults < `skala.toml` < command-line flags.

## Outputs and scratch space

- Default output name is `<name>_skala_<scale>x.<ext>`, next to the input.
- An existing output is an error unless you pass `--overwrite` (or pick another
  path with `-o`). In a folder run, that file is reported and the rest continue.
- Video work happens in a scratch directory (extracted frames live there). It is
  removed on success, and **kept** on failure so you can inspect it. `--keep-temp`
  keeps it always.
- Scratch location: `--workdir` if given, else `$SKALA_WORKDIR`, else the system
  temp directory.

## External tools

Skala finds the engine in this order, first hit wins:

1. `--engine-path`
2. the `$SKALA_ENGINE` environment variable
3. the managed install from `skala setup` (`<data dir>/engine`)
4. a `bin/` folder next to where you run Skala
5. your `PATH`

The engine binary is called `realesrgan-ncnn-vulkan` (`realesrgan-ncnn-vulkan.exe`
on Windows). Its `models/` folder is used automatically when it sits next to the
binary; `--model-dir` points at one somewhere else. If no engine is found you get
a clear error naming the missing tool rather than a stack trace. ffmpeg and
ffprobe are resolved the same way via `--ffmpeg-path` / `--ffprobe-path` and
`PATH`.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | A Skala error (missing input, missing tool, engine failure, ...) |
| `2` | Bad command-line usage (unknown option, wrong value type) |
| `130` | Interrupted with Ctrl+C |

When upscaling a folder, a failure on one file is reported and the rest continue;
the run still exits `1` so scripts can detect it.

## Known limitations

- Skala trains and ships no models; `skala setup` only downloads the engine
  release, which bundles its own `models/` folder.
- `--scale` must follow the input path on the command line.
- `--version` only works at the top level (`skala --version`), not after `run`.
- A file literally named `doctor` is ambiguous with the `doctor` subcommand —
  rename it or use `skala run doctor`.
- Video is re-encoded with ffmpeg (`libx264` by default); there is no streaming
  or GPU encoder selection yet.
- 8x is memory-hungry, especially on video.

## Development

```console
$ python -m pip install -e ".[dev]"
$ python -m pytest          # test suite
$ python -m ruff check .
$ python -m ruff format --check skala tests
$ python -m mypy skala tests
```

The external binaries are not required to run the tests: subprocess calls are
mocked, so the suite runs on any machine.

There is a small end-to-end suite (marked `integration`) that drives the real
engine and ffmpeg instead of mocks. It is opt-in and skips itself when either
tool is missing:

```console
$ SKALA_RUN_INTEGRATION=1 python -m pytest -m integration
```

On Windows PowerShell:

```console
$ $env:SKALA_RUN_INTEGRATION=1; python -m pytest -m integration
```

CI runs the mocked suite on Linux and Windows across Python 3.10-3.13, plus a
best-effort integration job on Linux.

## License

MIT
