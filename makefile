all : test_perceptron

test_perceptron : main.cpp perceptron.o 
	g++ -Wall -o test_perceptron main.cpp perceptron.o 

perceptron.o: perceptron.cpp perceptron.hpp
	g++ -Wall -c perceptron.cpp

clean:
	rm *.o test_perceptron 

