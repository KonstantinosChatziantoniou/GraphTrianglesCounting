#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
//https://proofwiki.org/wiki/Product_of_Triangular_Matrices

int max_per_row = 0;
__global__
void devTrianglesCount(int* col_indx,  int* csr_rows, int nnz, int rows, int* out_sum);

/** 
 * Description: Reads the data from the mtx files.
 * The first row contains 3 integers: rows columns of the sparse graph 
 * and the number of non zero elements. The non zero elements are stored in 
 * COO format. Also the data have one-based indexing. While reading them we tra them
 * to zero based indexing.
 * 
 * @param data char[] the name of the file to read
 * @param row_indx int*  where the rows of the nnz are stored
 * @param col_indx int*  where  the columns of the nnz are stored
 * @param nnz int* the number of non zero elements
 * @param rows int* the number of rows
 * @param cols itn* the number of columns
 */
void readData(char data[], int **row_indx, int **col_indx, int* nnz, int * rows, int* cols){
    FILE *f = fopen(data,"r");
    fscanf(f, "%d %d %d\n",rows, cols, nnz);
    printf("-READ %d %d %d\n",*rows,*cols,*nnz);
    col_indx[0] = (int*)malloc((*nnz)*sizeof(int));
    row_indx[0] = (int*)malloc((*nnz)*sizeof(int));
    for(int i = 0; i < *nnz; i++){
        fscanf(f, "%d %d", &col_indx[0][i] , &row_indx[0][i]);
        // data have 1 base index
        // transform to 0-based index
        col_indx[0][i]--;
        row_indx[0][i]--;
    }

    fclose(f);
}

/**
 * Description: Returns an array with the non zero rows in compressed format: (length rows insteadn of nnz).
 * Combined with the column index we have the CSR represantion of the sparse graph. Also finds the max non zero
 * elements per row and updates the global variable max_per_row
 * 
 * @param rows int 
 * @param nnz int
 * @param row_indx int* the row vector from the COO format.
 * 
 * 
 * Returns:
 *         csr_rows int*
 */
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
        printf("Invalid arguments\n");
        return 1;
    }



    //cudaDeviceReset();
    struct timeval start,end,ALLSTART,ALLEND;

    // "auto.mtx"; // "data.csv"; //  "great-britain_osm.mtx"; // "delaunay_n22.mtx"; //
    printf("-Dataset: %s\n",argv[1]);
    int rows,cols,nnz;
    int *col_indx, *row_indx;
    int sum;

    /* Read Data in COO format and transform to 0 based index */
    gettimeofday(&start,NULL);
    readData(argv[1],&row_indx,&col_indx,&nnz,&rows,&cols);
    gettimeofday(&end,NULL);
    printTime(start,end, "Read Data");


        
    // Transform to CSR
    gettimeofday(&start,NULL);
    int* csr_rows = COOtoCSR(rows, nnz, row_indx);
    // We no longer need row_indx since we have csr_rows
    free(row_indx);
    gettimeofday(&end,NULL);
    printTime(start,end, "CSR");
    
    printf("-MAX PER ROW = %d\n",max_per_row);
    
    gettimeofday(&start,NULL);
    cudaError_t cuer;
    int *cu_col_indx,  *cu_csr_rows;
    int* cu_sum;
    cuer = cudaMalloc(&cu_col_indx,nnz*sizeof(int));
    printf("-%s\n",cudaGetErrorName(cuer));
    cuer = cudaMalloc(&cu_csr_rows,rows*sizeof(int));
    printf("-%s\n",cudaGetErrorName(cuer));
    cuer = cudaMalloc(&cu_sum,rows*sizeof(int));
    printf("-%s\n",cudaGetErrorName(cuer));
    

    cuer = cudaMemcpy(cu_col_indx,col_indx,nnz*sizeof(int),cudaMemcpyHostToDevice);
    printf("-%s\n",cudaGetErrorName(cuer));
    cuer = cudaMemcpy(cu_csr_rows,csr_rows,rows*sizeof(int),cudaMemcpyHostToDevice);
    printf("-%s\n",cudaGetErrorName(cuer));

    int* res = (int*)malloc(rows*sizeof(int));
    for(int i = 0; i < rows; i++){
        res[i] = 0;
    }

    cudaMemcpy(cu_sum,res,rows*sizeof(int),cudaMemcpyHostToDevice);

    gettimeofday(&end,NULL);
    printTime(start,end, "CUDA data transfer");


    gettimeofday(&start,NULL);
    int threads = max_per_row;
    if(max_per_row > 250){
        return 1;
    }

    int blcoksize = rows/(512*512) + 1;
    printf("-blocksize %d %d\n", blcoksize, 512*512);
    devTrianglesCount<<<dim3(512,512,blcoksize),threads>>>(cu_col_indx, cu_csr_rows, nnz, rows, cu_sum);
    printf("-%s\n",cudaGetErrorName(cuer));




    cuer = cudaMemcpy(res,cu_sum,rows*sizeof(int),cudaMemcpyDeviceToHost);
    printf("-%s\n",cudaGetErrorName(cuer));


    sum = 0;
    for(int i = 0; i < rows; i++){
        
        sum += res[i];
    }

    printf("-Cuda triangles = %d\n",sum);
    gettimeofday(&end,NULL);
    printTime(start,end,"CUDA");
    
}


__global__
void devTrianglesCount(int* col_indx, int* csr_rows, int nnz, int num_of_rows, int* out_sum){
    int row = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
    int id = threadIdx.x;

    if(row >= num_of_rows){
        return;
    }
    __shared__ int start_row;
    __shared__ int end_row;
    __shared__ int len;
    __shared__ int* row_ptr;
    __shared__ int current_row[64];

    __shared__ int sh_len[64];
    __shared__ int* sh_ptr[64];
    __shared__ int sh_cols[64][64];
    __shared__ int sh_sum[64];
    sh_sum[id] = 0;
    __syncthreads();
    // Get the current row
    if(id == 0){
        start_row = csr_rows[row];
        if(row == num_of_rows-1){
            end_row = nnz;
        }else{
            end_row = csr_rows[row+1];
        }
        len = end_row - start_row;
        row_ptr = &col_indx[start_row];
    }
    __syncthreads();
    if(id < len){
        current_row[id] = row_ptr[id];
    }
    __syncthreads();

    // Get info for each column
    if(id < len){
        int tmp_col = current_row[id];
        int tmp_start = csr_rows[tmp_col];
        int tmp_end;
        if(tmp_col == num_of_rows-1){
            tmp_end = nnz;
        }else{
            tmp_end = csr_rows[tmp_col+1];
        }
        sh_len[id] =  tmp_end - tmp_start;
        sh_ptr[id] = &col_indx[tmp_start];
    }

    __syncthreads();
    for(int i = 0; i < len; i++){
        if(id < sh_len[i]){
            sh_cols[i][id] = sh_ptr[i][id];
        }
    }
    __syncthreads();

    if(id < len){
        int a = 0;
        int b = 0;
        int sum = 0;
        while(1){
            if(a == len || b == sh_len[id]){
                break;
            }

            int b1 = current_row[a] == sh_cols[id][b];
            int b2 = current_row[a] > sh_cols[id][b];
            int b3 = current_row[a] < sh_cols[id][b];

            a = a + b1 + b3;
            b = b + b1 + b2;
            sum = sum + b1;
        }

        sh_sum[id] = sum;
    }
    __syncthreads();
    if(id == 0){
        int sum = 0;
        for(int i = 0; i < len; i++){
            sum += sh_sum[i];
        }
        out_sum[row] = sum;
    }
    __syncthreads();
    
}
