# cuda-batch-fft-filter

CUDA Advanced Libraries capstone project that filters batches of signal CSV files
with cuFFT.

The program processes many one-dimensional signal files in a single batched GPU
FFT pipeline. If no input files exist, it generates synthetic noisy signals,
runs a forward cuFFT transform, attenuates high-frequency bins with a CUDA
kernel, runs an inverse cuFFT transform, and writes filtered CSV outputs plus a
summary report.

## Why This Meets The Assignment

- Uses an advanced CUDA library: cuFFT.
- Uses a custom CUDA kernel between cuFFT stages.
- Processes many small inputs in one run.
- Accepts command line arguments.
- Includes a `Makefile` and `run.sh`.
- Produces proof artifacts in `output/` and a zip file in `artifacts/`.

## Build

Run inside the Coursera CUDA lab or another machine with CUDA, cuFFT, and
`nvcc`:

```bash
make clean build
```

## Run

```bash
./run.sh
```

Equivalent manual command:

```bash
./batch_fft_filter --input data/input --output output/filtered --limit 100 --generate 100 --length 2048 --cutoff 96 --attenuation 0.05
```

Quick run without creating the proof zip:

```bash
make run
```

Run with custom arguments and still create proof artifacts:

```bash
./run.sh --input data/input --output output/filtered --limit 25 --generate 25 --length 1024 --cutoff 48 --attenuation 0.10
```

## CLI Arguments

- `--input`: directory containing input CSV signal files
- `--output`: directory for filtered CSV files
- `--limit`: maximum number of input files to process
- `--generate`: number of synthetic input files to generate if none exist
- `--length`: number of samples per generated signal
- `--cutoff`: frequency-bin cutoff to keep before attenuation
- `--attenuation`: multiplier applied to bins above the cutoff

## Output Artifacts

After running, the project writes:

- `output/run.log`: terminal log from `run.sh`
- `output/summary.csv`: per-file processing statistics
- `output/filtered/*.csv`: filtered signal outputs
- `output/proof-input/*.csv`: generated proof input signals used by `run.sh`
- `artifacts/cuda-batch-fft-filter-proof.zip`: compressed proof bundle

The summary file includes the input file, output file, number of samples,
frequency cutoff, attenuation value, roughness before filtering, roughness after
filtering, total batch GPU time, and estimated GPU time per file.

`run.sh` uses `output/proof-input/` for generated proof data so user-provided
files in `data/input/` are not removed during artifact generation.

This local copy may only contain placeholder artifact folders until the project
is run inside a CUDA environment. Run `./run.sh` in the Coursera lab to generate
the real proof files.

Example terminal log shape:

```text
Generating 100 input signals with 2048 samples each.
Loaded 100 signals with 2048 samples each.
Running batched cuFFT low-pass filter with cutoff 96 and attenuation 0.05.
processed=signal_0000.csv samples=2048 roughness_before=... roughness_after=... gpu_ms_per_file=...
Processed 100 signal files using batched cuFFT frequency filtering.
Summary: output/summary.csv
Filtered signals: output/filtered
```

## GPU Algorithm

1. Read many signal CSV files into one flattened host buffer.
2. Copy the batch into GPU memory as `cufftComplex` values.
3. Execute one batched forward `cufftExecC2C` transform.
4. Launch `ApplyLowPassFilter` to attenuate frequency bins above the cutoff.
5. Execute one batched inverse `cufftExecC2C` transform.
6. Launch `NormalizeInverseFft` because cuFFT inverse transforms are unscaled.
7. Copy the filtered batch back to host memory and write one output CSV per
   input file.

## Repo Structure

```text
.
├── Makefile
├── README.md
├── run.sh
├── src/main.cu
├── data/input/
├── output/proof-input/
├── output/filtered/
└── artifacts/
```

## Presentation Outline

1. Show the generated dataset in `data/input/`.
2. Explain why frequency-domain filtering is a good fit for cuFFT.
3. Walk through the forward FFT, CUDA frequency filter, inverse FFT, and
   normalization steps.
4. Run `./run.sh`.
5. Show `output/run.log`, `output/summary.csv`, and a few filtered CSV files.
6. Explain the roughness metric: a lower output roughness means adjacent samples
   changed less after high-frequency attenuation.

## Notes

The project intentionally avoids CPU-only processing for the main workload. The
CPU is used for CSV I/O and simple summary metrics; the signal transform and
frequency filtering run on the GPU through cuFFT and a CUDA kernel.
