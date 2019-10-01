# GraphTrianglesCounting

Here lies the code for the 4th assignments of the Parallel And Distributes Systems course.

There are 3 programs

      1. cuda  The implementation of the algorithm in CUDA with bad wrap activity efficiency
      2. cuda_wp The implementation of the algorithm in CUDA with improved wrap activity efficiency
      3. serial_cilk  The serial and parallel implemantation of the algorithm, combined in one executable
      
To compile eaech program you have to run make from each directory.
BEWARE! serial_cilk implementation uses cilk, so you have to change the header to include to point to the cilk library.

The programs take as first argument the dataset. 
BEWARE! The datasets have to be cleaned of the comments first(sry).

The scripts were only used for the plots and not automation.
