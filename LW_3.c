#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>



#define EPSILON (0.000244140625)
#define DEL_X (0.5)
#define DEL_Y (0.25)
#define ALPHA (1.25)
#define N (4)

#define max(a, b) ((a) >= (b) ? (a) : (b))

void file_write(const size_t m_x, const size_t m_y,
    double (* const  u_k)[m_y], const char * const  filename)
{
    int int_return;
    FILE * const file_out = fopen(filename, "wb");
    const size_t m_x_minus_1 = m_x - 1, m_y_minus_2 = m_y - 2U;
    for (size_t i = 1; i < m_x_minus_1; ++i)
    {
        size_t size_return = fwrite(u_k[i] + 1, sizeof(double), m_y_minus_2, file_out);
    }
    int_return = fclose(file_out);
}


void finalize(void * const  u_k, void * const  r)
{
    free(u_k);
    free(r);
}

void init(const size_t m_x, const size_t m_y,
    double (** const  u_k)[m_y],
    double (** const  r)[m_y])
{
    *u_k        = (double (*)[m_y]) malloc(m_x * m_y * sizeof(double));
    *r          = (double (*)[m_y]) malloc(m_x * m_y * sizeof(double));

    #pragma omp parallel for num_thread(N)
    for (size_t i = 0; i < m_x; ++i)
    {
        for (size_t j = 0; j < m_y; ++j)
        {
            if (!i || i + 1 == m_x || !j || j + 1 == m_y)
            {
                (*u_k)[i][j] = 1.0;
            }
            else
            {
                (*u_k)[i][j] = 0.0;
            }
            (*r)[i][j] = 0.0;
        }
    }
}


void start(const size_t m_x, const double del_x, const size_t m_y, const double del_y,
    const double epsilon, const char * const  filename)
{
    double (*  u_k)[m_y], (*  r)[m_y];
    init(m_x, m_y, &u_k, &r);
    const double rdx2 = 1.0 / del_x / del_x, rdy2 = 1.0 / del_y / del_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));
    const size_t m_y_minus_1 = m_y - 1;
    double norm[2U] = { epsilon }, wtime_start = omp_get_wtime(), wtime_end;

        for (size_t k = 0; max(norm[0], norm[1]) >= epsilon; k ^= 1)
        {

            norm[k] = 0.0;
            double norm_k = 0.0;
            #pragma omp parallel for private(norm_k) num_thread(N)
            for (size_t i = 1; i < m_x-1; ++i)
            {
                for (size_t j = 1 + (i + k) % 2U; j < m_y_minus_1; j += 2U)
                {
                    const double u_k_i_j = u_k[i][j];
                    u_k[i][j] = ((u_k[i - 1][j] + u_k[i + 1][j]) * rdx2 +
                        (u_k[i][j - 1] + u_k[i][j + 1]) * rdy2 - r[i][j]) * beta;
                    u_k[i][j] = u_k_i_j +(u_k[i][j]-u_k_i_j)*ALPHA;
                    norm_k = max(norm_k, fabs(u_k_i_j - u_k[i][j]));
                }
            }
            norm[k] = max(norm[k], norm_k);
        }
    wtime_end = omp_get_wtime();
    printf("wtime = %lf\n", wtime_end - wtime_start);
    file_write(m_x, m_y, u_k, filename);
    finalize(u_k, r);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <m rows> <m columns> <file out>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const ptrdiff_t m_x = strtoll(argv[1], NULL, 10) + 2,
        m_y = strtoll(argv[2], NULL, 10) + 2;
    start(m_x, DEL_X, m_y, DEL_Y, EPSILON, argv[3]);
    return 0;
}
