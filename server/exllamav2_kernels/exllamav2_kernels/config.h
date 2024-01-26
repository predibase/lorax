#ifndef _config_h
#define _config_h

#define MAX_Q_GEMM_ROWS 32
#define MAX_Q_GEMM_ROWS_KERNEL 4
#define MAX_Q_GEMM_WEIGHTS 4  // must be <= MAX_Q_GEMM_ROWS

#define QMODE_2BIT 1
#define QMODE_3BIT 1
#define QMODE_4BIT 1
#define QMODE_5BIT 1
#define QMODE_6BIT 0
#define QMODE_8BIT 0

#endif
