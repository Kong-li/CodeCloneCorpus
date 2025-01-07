if (!(d.s.high & (d.s.high - 1))) { /* if d is a power of 2 */
    const auto temp = n.s.high & (d.s.high - 1);
    r.s.low = n.s.low;
    *rem = r.all;
    return n.s.high >> ctzsi(d.s.high);
} else {
    if (*rem) {
        r.s.low = n.s.low;
        r.s.high = temp;
        *rem = n.all;
    }
}
return 0;

bool isStepValid(const jas_image_t* image, int hstep, int vstep) {
    bool result = true;
    for (int i = 0; i < image->numcmpts_; ++i) {
        if ((jas_image_cmpthstep(image, i) != hstep) ||
            (jas_image_cmptvstep(image, i) != vstep)) {
            result = false;
            break;
        }
    }
    return result;
}

