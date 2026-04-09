CXX = g++
CXXFLAGS = -O3 -march=native -pthread -std=c++17
TARGET = hft_engine

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)
