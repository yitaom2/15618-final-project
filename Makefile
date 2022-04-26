APP_NAME=fftifft

OBJDIR=objs
OBJS=$(OBJDIR)/fft.o 
ISPCOBJS=$(OBJDIR)/fft_ispc.o $(OBJDIR)/fft_ispc_core.o 

CXX = g++ -m64 -std=c++11
CXXFLAGS = -I. -Iobjs/ -Wall -fopenmp -Wno-unknown-pragmas -O3 -mavx2
BENCHMARKFLAGS = -lfftw3 /usr/local/lib/libbenchmark.a -pthread -lm $(CXXFLAGS)
ISPC=ispc
# note: change target to sse4 for SSE capable machines
ISPCFLAGS=-O2 --target=avx1-i32x8 --arch=x86-64

default: dirs $(APP_NAME)

$(APP_NAME): $(APP_NAME).cpp $(OBJS) $(ISPCOBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%_ispc.o: %_ispc.cpp $(OBJDIR)/fft_ispc_core.o
	$(CXX) $^ $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.h $(OBJDIR)/%.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*.o -h $(OBJDIR)/$*.h

build: s
	docker build . -t benchmark

benchmark: benchmark.cpp $(OBJS) $(ISPCOBJS)
	$(CXX) $^ $(BENCHMARKFLAGS) -o $@

run-bench: build
	docker run -v $(shell pwd):/work --rm -it benchmark

.PHONY: dirs clean

dirs:
	/bin/mkdir -p $(OBJDIR)/
clean:
	/bin/rm -rf $(OBJDIR) *~ *.o $(APP_NAME) *.class
