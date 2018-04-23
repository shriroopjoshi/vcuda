CC = g++
CFLAGS = -std=c++11
SRC = vcuda_client.cpp
INCLUDE = -I../include/
OBJ = $(SRC:.cpp = .o)
LIBNAME = libvcuda.a

.PHONY: all clean build lib

all: lib

build: $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDE) -o vcuda $(OBJ)

clean:
	rm -f *.o *.out
	rm -f vcuda
	rm -f $(LIBNAME)

lib: clean
	$(CC) $(CFLAGS) $(INCLUDE) -c $(SRC)
	ar rcs $(LIBNAME) *.o