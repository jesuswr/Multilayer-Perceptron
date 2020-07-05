all : test_perceptron test_perceptron2 test_perceptron3

test_perceptron : main.cpp perceptron.o multilayer_perceptron.o
	g++ -Wall -o test_perceptron main.cpp perceptron.o multilayer_perceptron.o

test_perceptron2 : main2.cpp perceptron.o multilayer_perceptron.o
	g++ -Wall -o test_perceptron2 main2.cpp perceptron.o multilayer_perceptron.o

test_perceptron3 : main3.cpp perceptron.o multilayer_perceptron.o
	g++ -Wall -o test_perceptron3 main3.cpp perceptron.o multilayer_perceptron.o

perceptron.o: perceptron.cpp perceptron.hpp
	g++ -Wall -c perceptron.cpp

multilayer_perceptron.o: multilayer_perceptron.cpp multilayer_perceptron.hpp
	g++ -Wall -c multilayer_perceptron.cpp

clean:
	rm *.o test_perceptron 

