all: main.cpp
	g++ -std=c++11 main.cpp -o build/main.o
	./build/main.o
debug: main.cpp
	g++ -g -std=c++11 main.cpp -o build/main.d
	./build/main.d