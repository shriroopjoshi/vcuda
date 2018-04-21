CC = g++
CFLAGS = -std=c++11
SRC = vcuda.cpp vcuda_client.cpp
INCLUDE = -I../include/
OBJ = $(SRC:.cpp = .o)

.PHONY: all clean build

all: clean build

build: $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDE) -o vcuda $(OBJ)

clean:
	rm -f *.o
	rm -f vcuda