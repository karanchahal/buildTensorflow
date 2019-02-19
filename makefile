all: main.cpp
	g++ -std=c++11 main.cpp -o build/main.o
	./build/main.o