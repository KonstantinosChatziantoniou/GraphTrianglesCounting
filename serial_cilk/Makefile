CC = gcc
CFLAGS = -fcilkplus -Ofast -std=c99
LIBS = -lm -Wall -Wextra -lblas 
OBJDIR = ./bin
HDRDIR = ./headers
SRCDIR = ./src


##CHANGE HERE
_OBJ =  main.o qsortSec.o
OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))




##CHANGE HERE
_DEPS = qsortSec.h
DEPS = $(patsubst %, $(HDRDIR)/%, $(_DEPS))


mainProgram: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $<  $(CFLAGS)

clean:
	rm -rf ./*.csv