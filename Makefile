NVCC ?= nvcc
NVCCFLAGS ?= -std=c++14 -O2
CUDA_LIBS ?= -lcufft
RUN_ARGS ?= --input data/input --output output/filtered --limit 100 --generate 100 --length 2048 --cutoff 96 --attenuation 0.05

.PHONY: clean build run proof

build: src/main.cu
	$(NVCC) $(NVCCFLAGS) src/main.cu -o batch_fft_filter $(CUDA_LIBS)

run: build
	./batch_fft_filter $(RUN_ARGS)

proof: build
	mkdir -p output/filtered artifacts
	./batch_fft_filter $(RUN_ARGS) | tee output/run.log
	if command -v zip >/dev/null 2>&1; then zip -qr artifacts/cuda-batch-fft-filter-proof.zip README.md Makefile run.sh src/main.cu data/input output; fi

clean:
	rm -f batch_fft_filter
	rm -f output/run.log output/summary.csv output/filtered/*.csv
	rm -f artifacts/cuda-batch-fft-filter-proof.zip
