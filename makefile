all: main.cpp
	g++ -std=c++11 main.cpp -o build/main.o
	./build/main.o
debug: main.cpp
	g++ -g -std=c++11 main.cpp -o build/main.o
	./build/main.o
test: test.cpp
	g++ -std=c++11 test.cpp -lgtest -lgtest_main -pthread -o build/test.o
	./build/test.o
gpuTest: cudaOps.cu
	nvcc -std=c++14 cudaOps.cu -o build/cuda.o
	./build/cuda.o
