CC = g++
CFLAGS = -std=c++11
INCLUDE = -I../
LDPATH = -L../
LIBS = -lvcuda

all: clean example

example: example.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@ $(LDPATH) $(LIBS)

clean:
	rm -f example