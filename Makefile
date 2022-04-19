APP_NAME=fft

OBJS=fft.o

CXX = g++ -m64 -std=c++11
CXXFLAGS = -I. -O3 -Wall -fopenmp -Wno-unknown-pragmas
BENCHMARKFLAGS = -lfftw3 /usr/local/lib/libbenchmark.a -pthread -lm

default: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

benchmark: benchmark.cpp
	$(CXX) $< $(BENCHMARKFLAGS) -o $@

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class
