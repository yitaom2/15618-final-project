APP_NAME=fftifft

OBJS=fft.o

CXX = g++ -m64 -std=c++11
CXXFLAGS = -I. -Wall -fopenmp -Wno-unknown-pragmas -g
BENCHMARKFLAGS = -lfftw3 /usr/local/lib/libbenchmark.a -pthread -lm -Wall -fopenmp -Wno-unknown-pragmas -g

default: $(APP_NAME)

$(APP_NAME): $(APP_NAME).cpp $(OBJS)
	$(CXX) $< $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

benchmark: benchmark.cpp $(OBJS)
	$(CXX) $< $(BENCHMARKFLAGS) -o $@ $(OBJS)

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class
