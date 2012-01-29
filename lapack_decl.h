/*
Copyright (c) 2012, Victor Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _LAPACK_DECL_H_
#define _LAPACK_DECL_H_

#include <stdlib.h>

// These are function prototypes for the LAPACK routines that lp_tiny
// calls.

// AMD Core Math Library requires a significant amount of accomodation.
// None of the LAPACK routines require workspace parameters, and several
// BLAS routines do not enforce const-correctness.

# define FCALL(name) name ## _

double FCALL(ddot)(const int* N, 
                   const double* X, const int* incX, 
                   const double* Y, const int* incY);

double FCALL(dnrm2)(const int* N, 
                    const double* X, const int* incX);

void FCALL(dcopy)(const int* N,
                  const double* X, const int* incX,
                  double* Y, const int* incY);

void FCALL(daxpy)(const int* N,
                  const double* alpha,
                  const double* X, const int* incX,
                  double* Y, const int* incY);

void FCALL(dscal)(const int* N,
                  const double* alpha,
                  double* X, const int* incX);

void FCALL(dgemv)(const char* trans, const int* M, const int* N,
                  const double* alpha,
                  const double* A, const int* lda,
                  const double* X, const int* incX,
                  const double* beta,
                  double* Y, const int* incY);

void FCALL(dgemm)(const char* transA, const char* transB,
                  const int* M, const int* N, const int* K,
                  const double* alpha,
                  const double* A, const int* lda,
                  const double* B, const int* ldb,
                  const double* beta,
                  double* C, const int* ldc);

void FCALL(dgels)(const char *trans, const int *m, const int *n,
                  const int *nrhs, double *a, const int *lda,
                  double *b, const int *ldb,
                  double *work, const int *lwork, int *info);

// SVD is only needed in lp_tiny_generate
void FCALL(dgesvd)(const char *jobu, const char *jobvt,
                   const int *m, const int *n,
                   double *a, const int *lda, double *s,
                   double *u, const int *ldu, double *vt, const int *ldvt,
                   double *work, const int *lwork, int *info);

void FCALL(dsysv)(const char *uplo, const int *n, const int *nrhs,
                  double const *a, const int *lda, int *ipiv,
                  double *b, const int *ldb,
                  double *work, const int *lwork, int *info);

void FCALL(dgesv)(const int *n, const int *nrhs,
                  double *a, const int *lda, int *ipiv,
                  double *b, const int *ldb, int *info);

void FCALL(dsyr2k)(const char* uplo, const char* trans,
                   const int* N, const int* K,
                   const double* alpha,
                   const double* A, const int* lda,
                   const double* B, const int* ldb,
                   const double* beta,
                   double* C, const int* ldc);

#endif // _LAPACK_DECL_H_
