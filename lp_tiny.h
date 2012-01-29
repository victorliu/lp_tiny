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

#ifndef LP_TINY_H_INCLUDED
#define LP_TINY_H_INCLUDED

#include <stdio.h>

// Written by Victor Liu for EE364a @ Stanford
// Thanks Prof. Boyd for a wonderful class!

// Represents a standard form linear program
//     minimize   (c' * x)
//     subject to A*x == b, x >= 0 (elementwise)
// x is the variable, of dimension n. A has m rows, with m < n.
// It is assumed that A is full rank, and that the sublevel sets
// {x | A*x == b, x >= 0, c'*x <= gamma} are all bounded.
//
// Note: If you know your problem is bounded but lp_tiny is
// saying that it is unbounded, this probably means you have
// unbounded sublevel sets. You need to add in an extra constraint
// (e.g. the sum of all your variables is less than some number).

typedef struct lp_tiny_struct{
	int n, m;
	double *A, *b, *c;
	// Note that A is assumed to be in column-major (Fortran) order.
	// i.e. A_{i,j} is located at A[i+j*m]
	// Also, A is allocated with an extra column to fascilitate
	// implementation of lp_solve()
} lp_tiny;

// Returns negative number for errors in argument (-value is which argument)
// Returns positive number for allocation error
// Returns 0 on success
int  lp_tiny_init(lp_tiny *lp, int n, int m);
void lp_tiny_destroy(lp_tiny *lp);

// Prints out an equivalent block of Matlab code that solves the same
// problem in CVX.
void lp_tiny_print(lp_tiny *lp, FILE *fp);

void lp_tiny_print_vector(int n, double *v, FILE *fp);
void lp_tiny_print_matrix(int rows, int cols, double *m, int ldm, FILE *fp);

double lp_tiny_eval(const lp_tiny *lp, double *x); // evaluates the objective function
// Returns nonzero if in domain of inequality constraints
int lp_tiny_in_domain(const lp_tiny *lp, double *x);

typedef enum lp_tiny_status_enum{
	LP_TINY_SOLVED = 0,
	LP_TINY_INFEASIBLE, // Actually, just not strictly feasible
	LP_TINY_UNBOUNDED,  // Unbounded sublevel sets
} lp_tiny_status;
// Converts a status enum into a string for printing.
const char *lp_tiny_status_string(lp_tiny_status status);

// Purpose:
//   Solves the LP
//     minimize   c'*x
//     subject to A*x == b, x >= 0 (element-wise)
//   with variable x of length n, and m equality constraints with m < n
//   and also assuming the problem has bounded sublevel sets.
// Inputs:
//   lp     : Linear program with full rank A (nonsingular KKT matrix)
// Outputs:
//   x      : Primal optimal point (assumes preallocated)
//   lambda : Optional dual variable to inequality constraints
//   nu     : Optional dual variable to equality constraints
//   status : The solution status; only valid if this function returned 0
//            If the problem is not strictly feasible, LP_INFEASIBLE is
//            returned. Does not detect unbounded problems, so LP_UNBOUNDED
//            is never returned.
// Return value:
//   0      : Success
//   < 0    : Error in a parameter. The negative gives which one.
//   1      : Memory allocation error
//   2      : Internal maximum iteration limit reached, convergence failed.
//   3      : The A matrix is not full rank.
//   4      : Line search step size went to zero.
//   5      : Failed least norm solution of A*xt == b.
int lp_tiny_solve(
	const lp_tiny *lp,
	double *x,
	double *lambda, double *nu,
	lp_tiny_status* status, int *n_steps
);

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Private methods ///////////////////////////////
/////// (but you can call them if you think you know what you're doing) ///////
///////////////////////////////////////////////////////////////////////////////

// Purpose:
//   Uses Newton's method to solve the centering problem
//     minimize   c'*x - sum_{i=0}^n log x_i
//     subject to A*x == b
//   with variable x of length n, given a strictly feasible starting point x0.
// Inputs:
//   lp        : Linear program with full rank A (nonsingular KKT matrix)
//   x0        : Initial strictly feasible starting point
//   barrier_t : The (inverse) weight of the barrier:
//                 f0(x) = barrier_t * c'*x - sum_{i=0}^n log x_i
//               is the function that is minimized.
//   iter_limit: Max number of iterations allowed.
// Outputs:
//   x_opt     : Primal optimal point (assumes preallocated)
//   nu_opt    : A dual optimal point (assumes preallocated)
//   n_steps   : The number of Newton steps if this parameter was not NULL.
// Return value:
//   0         : Success
//   < 0       : Error in a parameter. The negative gives which one.
//   1         : Memory allocation error
//   2         : Internal maximum iteration limit reached, convergence failed.
//   3         : An error occured in symmetric positive definite matrix solve.
int lp_tiny_centering_newton(
	const lp_tiny *lp,
	const double *x0,
	const double *barrier_t,
	int iter_limit,
	double *x_opt, double *nu_opt,
	int *n_steps
);

// Purpose:
//   Checks the solution x and the dual variable nu against the KKT condition.
// Return value:
//   0       : Valid solution
//  -1       : Invalid solution
//   1       : Memory allocation error
int lp_tiny_check_KKT(
	const lp_tiny *lp, double *x, double *nu
);

// Purpose:
//   Solves
//     minimize   c'*x
//     subject to A*x == b, x >= 0 (element-wise)
//   with variable x of length n, given a strictly feasible starting point x0.
// Inputs:
//   lp         : Linear program with full rank A (nonsingular KKT matrix)
//   x0         : Initial strictly feasible starting point
//   iter_limit : Max number of iterations allowed in the centering step.
// Outputs:
//   x_opt      : Primal optimal point (assumes preallocated)
//   lambda_opt : Optional dual variable to inequality constraints
//   nu_opt     : Optional dual variable to equality constraints
// Return value:
//   0   : Success
//   < 0 : Error in a parameter. The negative gives which one.
//   1   : Memory allocation error
//   2   : Internal maximum iteration limit reached, convergence failed.
int lp_tiny_solve_with_feasible_starting_point(
	const lp_tiny *lp,
	const double *x0,
	int iter_limit,
	double *x_opt,
	double *lambda_opt, double *nu_opt,
	int *n_steps
);

// Generate a test problem of size n and m. If feasible is nonzero, then
// the problem will have a initial feasible solution x0, otherwise the
// generated problem is infeasible. This function calls lp_tiny_init, so
// do not call it beforehand.
void lp_tiny_generate(lp_tiny *lp, int n, int m, int feasible, double *x0);

#endif // LP_TINY_H_INCLUDED
