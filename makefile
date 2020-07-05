all : test_perceptron

test_perceptron : main.cpp perceptron.o multilayer_perceptron.o
	g++ -Wall -o test_perceptron main.cpp perceptron.o multilayer_perceptron.o

perceptron.o: perceptron.cpp perceptron.hpp
	g++ -Wall -c perceptron.cpp

multilayer_perceptron.o: multilayer_perceptron.cpp multilayer_perceptron.hpp
	g++ -Wall -c multilayer_perceptron.cpp

clean:
	rm *.o test_perceptron 

