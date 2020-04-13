#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "cilk/cilk.h"

int max_per_row = 0;

int countParallel(int rows, int nnz, int* csr_rows, int* col_indx);

int countSerial(int rows, int nnz, int* csr_rows, int* col_indx);
//https://proofwiki.org/wiki/Product_of_Triangular_Matrices
void readData(char data[], int **row_indx, int **col_indx, int* nnz, int * rows, int* cols){
    FILE *f = fopen(data,"r");
    fscanf(f, "%d %d %d\n",rows, cols, nnz);
    printf("-READ %d %d %d\n",*rows,*cols,*nnz);
    col_indx[0] = malloc((*nnz)*sizeof(int));
    row_indx[0] = malloc((*nnz)*sizeof(int));
    for(int i = 0; i < *nnz; i++){
        fscanf(f, "%d %d", &col_indx[0][i] , &row_indx[0][i]);
        // data have 1 base index
        // transform to 0-based index
        col_indx[0][i]--;
        row_indx[0][i]--;
    }

    fclose(f);
}
int* COOtoCSR(int rows, int nnz, int* row_indx){
    // initialize
    int* csr_rows = (int*)malloc(rows*sizeof(int));
    for(int i = 0; i < rows; i++){
        csr_rows[i] = 0;
    }

    // Transformation to CSR 
    for(int i = 0; i < nnz; i++){
        int index = row_indx[i]+1;
        if(index < rows)
            csr_rows[index]++;
    }
    
    for(int i = 1; i < rows; i++){
        if(csr_rows[i] > max_per_row){
            max_per_row = csr_rows[i];
        }
        csr_rows[i] += csr_rows[i-1];
    }


    return csr_rows;
}

// int* CSRtoCSC(int rows, int nnz, int* row_indx, int* col_indx){
//     int* temp_col_indx = (int*)malloc(nnz*sizeof(int));
//     memcpy(temp_col_indx,col_indx,nnz*sizeof(int));
//     qsort_seq(temp_col_indx,row_indx,nnz);
//     free(temp_col_indx);
//     return COOtoCSR(rows,nnz,col_indx);
// }

void printTime(struct timeval start, struct timeval end, char* str){
    unsigned long ss,es,su,eu,s,u;
    ss  =start.tv_sec;
    su = start.tv_usec;
    es = end.tv_sec;
    eu = end.tv_usec;
    s = es - ss;
    if(eu > su){
        u = eu - su;
    }else{
        s--;
        u = 1000000 + eu - su;
    }
   
    printf("%s,%lu,%lu\n",str,s,u);
}
int main(int argc, char** argv){

     if(argc != 2){
        return 1;
    }
    struct timeval start,end,ALLSTART,ALLEND;

    // "auto.mtx"; // "data.csv"; //  "great-britain_osm.mtx"; // "delaunay_n22.mtx"; //
    printf("-Dataset: %s\n",argv[1]);
    int rows,cols,nnz;
    int *col_indx, *row_indx, *C_col_indx, *C_row_indx;
    /* Read Data in COO format and transform to 0 based index */
    gettimeofday(&start,NULL);
    readData(argv[1],&row_indx,&col_indx,&nnz,&rows,&cols);
    gettimeofday(&end,NULL);
    printTime(start,end, "Read Data");




    // Transform to CSR
    gettimeofday(&start,NULL);
    int* csr_rows = COOtoCSR(rows, nnz, row_indx);
    free(row_indx);
    gettimeofday(&end,NULL);
    printTime(start,end,"CSR");
    
    printf("-Max per row %d\n",max_per_row);

    int sum;    
    gettimeofday(&start,NULL);
    sum = countSerial(rows,nnz,csr_rows, col_indx);
    gettimeofday(&end,NULL);
    printf("-triangles %d\n",sum);
    printTime(start,end,"Serial");






    gettimeofday(&start,NULL);
    sum = countParallel(rows,nnz,csr_rows, col_indx);
    gettimeofday(&end,NULL);
    printTime(start,end,"Cilk");


   

    return 0;
}

int countParallel(int rows, int nnz, int* csr_rows, int* col_indx){
int sum;
    sum = 0;
    int* res = (int*)malloc(rows*sizeof(int));
    memset(res, 0, rows*sizeof(int));
    #pragma cilk_grainsize = rows / (4 * current_worker_count)
    cilk_for(int r = 0; r < rows; r++){
        int uplim;
        if(r == rows-1){
            uplim = nnz;
        }else{
            uplim = csr_rows[r+1];
        }

        int per_row = uplim - csr_rows[r];
        int* current_row = &col_indx[csr_rows[r]];

        for(int c = 0; c < per_row; c++){
            int current_col_index = current_row[c];
            int* current_col = &col_indx[csr_rows[current_col_index]];
            if(current_col_index == rows-1){
                uplim = nnz;
            }else{
                uplim = csr_rows[current_col_index+1];
            }
            int per_col = uplim - csr_rows[current_col_index];

            int a = 0;
            int b = 0;
            while(1){
                if(a == per_row || b == per_col){
                    break;
                }

                if(current_row[a] == current_col[b]){
                    res[r]++;
                    a++;
                    b++;
                    continue;
                }

                if(current_row[a] > current_col[b]){
                    b++;
                    continue;
                }

                if(current_row[a] < current_col[b]){
                    a++;
                    continue;
                }
            }
        }
    }

    for(int i = 0; i < rows; i++){
        sum += res[i];
    }
    free(res);
    return sum;
}


int countSerial(int rows, int nnz, int* csr_rows, int* col_indx){
    int sum;
    sum = 0;
    for(int r = 0; r < rows; r++){
        int uplim;
        if(r == rows-1){
            uplim = nnz;
        }else{
            uplim = csr_rows[r+1];
        }

        int per_row = uplim - csr_rows[r];
        int* current_row = &col_indx[csr_rows[r]];

        for(int c = 0; c < per_row; c++){
            int current_col_index = current_row[c];
            int* current_col = &col_indx[csr_rows[current_col_index]];
            if(current_col_index == rows-1){
                uplim = nnz;
            }else{
                uplim = csr_rows[current_col_index+1];
            }
            int per_col = uplim - csr_rows[current_col_index];

            int a = 0;
            int b = 0;
            while(1){
                if(a == per_row || b == per_col){
                    break;
                }

                if(current_row[a] == current_col[b]){
                    sum++;
                    a++;
                    b++;
                    continue;
                }

                if(current_row[a] > current_col[b]){
                    b++;
                    continue;
                }

                if(current_row[a] < current_col[b]){
                    a++;
                    continue;
                }
            }
        }
    }

    return sum;
}