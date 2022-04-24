APP_NAME=fftifft

OBJS=fft.o

CXX = g++ -m64 -std=c++11
CXXFLAGS = -I. -Wall -fopenmp -Wno-unknown-pragmas -O3
BENCHMARKFLAGS = -lfftw3 /usr/local/lib/libbenchmark.a -pthread -lm $(CXXFLAGS)

default: $(APP_NAME)

$(APP_NAME): $(APP_NAME).cpp $(OBJS)
	$(CXX) $< $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

build: 
	docker build . -t benchmark

benchmark: benchmark.cpp $(OBJS)
	$(CXX) $< $(BENCHMARKFLAGS) -o $@ $(OBJS)

run-bench: build
	docker run -v $(shell pwd):/work --rm -it benchmark

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class
