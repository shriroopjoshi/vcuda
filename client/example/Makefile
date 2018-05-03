CC = g++
CFLAGS = -std=c++11
INCLUDE = -I../
LDPATH = -L../
LIBS = -lvcuda

all: clean example.out add.out

example.out: example.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@ $(LDPATH) $(LIBS)

add.out: add.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $< -o $@ $(LDPATH) $(LIBS)

clean:
	rm -f *.out