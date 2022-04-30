APP_NAME=fftifft

OBJS=fft.o

CXX = g++ -m64 -mavx -std=c++11
CXXFLAGS = -I. -Wall -fopenmp -Wno-unknown-pragmas -O3
BENCHMARKFLAGS = -lfftw3 -I${HOME}/private/benchmark/include -I${HOME}/private/benchmark/build/include ${HOME}/private/benchmark/build/src/libbenchmark.a -pthread -lm $(CXXFLAGS)

LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
CU_FILES   := fft_cuda.cu
LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

default: $(APP_NAME)

$(APP_NAME): $(APP_NAME).cpp $(OBJS)
	$(CXX) $< $(CXXFLAGS) -o $@ $(OBJS)

cuda: fftifft_cuda.cpp fft_cuda.o
	$(CXX) $(CXXFLAGS) -o fftifft_cuda $^ $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

fft_cuda.o: fft_cuda.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

build: 
	docker build . -t benchmark

benchmark: benchmark.cpp $(OBJS)
	$(CXX) $< $(BENCHMARKFLAGS) -o $@ $(OBJS)

run-bench: build
	docker run -v $(shell pwd):/work --rm -it benchmark

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class
