NVCC ?= nvcc
NVCCFLAGS ?= -std=c++14 -O2
CUDA_LIBS ?= -lcufft
RUN_ARGS ?=
PROOF_ARGS ?= --input output/proof-input --output output/filtered

.PHONY: clean build run proof

build: src/main.cu
	$(NVCC) $(NVCCFLAGS) src/main.cu -o batch_fft_filter $(CUDA_LIBS)

run: build
	./batch_fft_filter $(RUN_ARGS)

proof: build
	mkdir -p output/proof-input output/filtered artifacts
	rm -f output/run.log output/summary.csv output/filtered/*.csv output/proof-input/*.csv artifacts/cuda-batch-fft-filter-proof.zip
	./batch_fft_filter $(PROOF_ARGS) $(RUN_ARGS) > output/run.log
	cat output/run.log
	zip -qr artifacts/cuda-batch-fft-filter-proof.zip README.md Makefile run.sh src/main.cu data output

clean:
	rm -f batch_fft_filter
	rm -f output/run.log output/summary.csv output/filtered/*.csv output/proof-input/*.csv
	rm -f artifacts/cuda-batch-fft-filter-proof.zip
