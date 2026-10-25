/* Two-pass CGS using a view of the first k columns, without copying V(:,1:k).
 * Uses MATLAB's BLAS and its configured computational-thread limit.
 * Build with mex -R2018a -O helm_gmres_cgs2_mex.c -lmwblas. */
#include "mex.h"
#include "blas.h"
#include <string.h>
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3 || nlhs != 2 || !mxIsDouble(prhs[0]) || mxIsSparse(prhs[0]) ||
        !mxIsDouble(prhs[1]) || mxIsSparse(prhs[1]) ||
        mxGetNumberOfElements(prhs[2]) != 1 || !mxIsDouble(prhs[2]))
        mexErrMsgIdAndTxt("helm:cgs2Input", "Expected [w,h]=f(dense double V,w,k).");
    ptrdiff_t n=(ptrdiff_t)mxGetM(prhs[0]), inc=1;
    double kd=mxGetScalar(prhs[2]);
    if (!isfinite(kd) || kd<1 || kd!=floor(kd) || kd>mxGetN(prhs[0]) ||
        mxGetNumberOfElements(prhs[1])!=(mwSize)n)
        mexErrMsgIdAndTxt("helm:cgs2Shape", "Invalid k or vector size.");
    ptrdiff_t k=(ptrdiff_t)kd;
    plhs[0]=mxCreateDoubleMatrix((mwSize)n,1,mxCOMPLEX);
    plhs[1]=mxCreateDoubleMatrix((mwSize)k,1,mxCOMPLEX);
    mxComplexDouble *w=mxGetComplexDoubles(plhs[0]);
    mxComplexDouble *h=mxGetComplexDoubles(plhs[1]);
    mxComplexDouble *hc=mxCalloc((mwSize)k,sizeof(mxComplexDouble));
    if (mxIsComplex(prhs[1])) {
        memcpy(w,mxGetComplexDoubles(prhs[1]),(size_t)n*sizeof(mxComplexDouble));
    } else {
        const double *input=mxGetDoubles(prhs[1]);
        for(ptrdiff_t i=0;i<n;++i) w[i].real=input[i];
    }
    /* MATLAB may store the first (real) Arnoldi column as a real matrix.
       Only that initialized prefix needs conversion, normally k=1. */
    mxComplexDouble *converted=NULL;
    const double *V;
    if (mxIsComplex(prhs[0])) {
        V=(const double *)mxGetComplexDoubles(prhs[0]);
    } else {
        const double *realV=mxGetDoubles(prhs[0]);
        converted=mxCalloc((mwSize)n*(mwSize)k,sizeof(mxComplexDouble));
        for(ptrdiff_t i=0;i<n*k;++i) converted[i].real=realV[i];
        V=(const double *)converted;
    }
    const double one[2]={1,0}, zero[2]={0,0}, minus[2]={-1,0};
    zgemv("C",&n,&k,one,V,&n,(double *)w,&inc,zero,(double *)h,&inc);
    zgemv("N",&n,&k,minus,V,&n,(double *)h,&inc,one,(double *)w,&inc);
    zgemv("C",&n,&k,one,V,&n,(double *)w,&inc,zero,(double *)hc,&inc);
    zgemv("N",&n,&k,minus,V,&n,(double *)hc,&inc,one,(double *)w,&inc);
    for(ptrdiff_t i=0;i<k;++i) {h[i].real+=hc[i].real;h[i].imag+=hc[i].imag;}
    mxFree(hc);
    if (converted) mxFree(converted);
}
