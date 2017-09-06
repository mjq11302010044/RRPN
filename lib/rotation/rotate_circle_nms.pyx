import numpy as np
cimport numpy as np

def a(np.float x,np.float r):
    return 0.5*3.1415926535*r*r-x*np.sqrt(r*r-x*x) - r*r*np.arcsin(x/r)


def rotate_cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x_ctrs = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y_ctrs = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] heights = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] widths = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 5]

    cdef np.ndarray[np.float32_t, ndim=1] rs = np.sqrt(heights**2+widths**2)/2.0
    cdef np.ndarray[np.float32_t, ndim=1] areas = 3.1415926535*rs*rs

    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix_ctr, iy_ctr, ir, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx_ctr, yy_ctr,rr
    cdef np.float32_t inter, ovr
    
    cdef np.float32_t d,x1,x2,s,r1,r2
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix_ctr = x_ctrs[i]
        iy_ctr = y_ctrs[i]
        ir = rs[i]
        iarea = areas[i]

        for _j in range(_i+1,ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx_ctr = x_ctrs[j]
            yy_ctr = y_ctrs[j]
            rr = rs[j]

            d = np.sqrt((ix_ctr-xx_ctr)**2+(iy_ctr-yy_ctr)**2)
            
            if ir<=rr:
                r1 = ir
                r2 = rr
            else:
                r1 = rr
                r2 = ir

            if d > 0.0:
                x1 = (d*d+r1*r1-r2*r2)/(2*d)
                x2 = (d*d+r2*r2-r1*r1)/(2*d)
                s = (r2*r2-r1*r1-d*d)/(2*d)
            #else: Avoid Warning
            #    x1 = 0
            #    x2 = 0
            #    s = 0

            if d<=r2-r1:
                inter = 3.1415926535*r1*r1
            elif d>=r2+r1 or r2 == 0 or r1 == 0:
                inter = 0.0
            else:
                if d*d<r2*r2-r1*r1:
                    inter = 3.1415926535*r1*r1-a(s,r1)+a(s+d,r2)
                else:
                    inter = a(x1,r1)+a(x2,r2)

            ovr = inter/(iarea+areas[j]-inter)
            #print i,j,ovr
            #print r1,r2,d
            if ovr>=thresh:
                suppressed[j]=1
    return keep
