all: main.cpp
	g++ -std=c++11 -I . main.cpp -o build/main.o
	./build/main.o
debug: main.cpp
	g++ -g -std=c++14 -I . main.cpp -o build/main.o
	./build/main.o
test: tests/main.cpp
	g++ -std=c++11 -I . tests/main.cpp -lgtest -lgtest_main -pthread -o build/test.o
	./build/test.o
gpuTest: gpu/cudaOps.cu
	nvcc -std=c++14 -I . gpu/cudaOps.cu -o build/cuda.o
	./build/cuda.o
gpu: main.cu
	nvcc -std=c++14 -I . main.cu -o build/cuda.o
	./build/cuda.o
	
