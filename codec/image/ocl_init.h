#ifndef _OCL_INIT_H
#define _OCL_INIT_H

#include <stdbool.h>
#include <CL/cl.h>

enum ocl_device_type {
    ODT_CPU, ODT_GPU, ODT_ACC
};

/*
 * Abort the program after printig an OpenCL error code.
 */
void error_and_abort(const char *msg, cl_int err);

/*
 * Find an OpenCL context and device (returned in 'devid').
 */
cl_context create_cl_context(enum ocl_device_type devtype, cl_device_id *devid);

/*
 * Load OpenCL kernel source from 'filename' and compile it
 * for the given context and device.
 */
cl_program build_program(const char *filename, cl_context context, cl_device_id devid);

/*
 * Return the start or end time in nano seconds of the given OpenCL event.
 */
cl_ulong get_event_start_nanos(cl_event event);
cl_ulong get_event_end_nanos(cl_event event);

#endif
