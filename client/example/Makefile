CC = g++
CFLAGS = -std=c++11
INCLUDE = -I../
LDPATH = -L../
LIBS = -lvcuda

all: clean example.out

example.out: example.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@ $(LDPATH) $(LIBS)

clean:
	rm -f *.out