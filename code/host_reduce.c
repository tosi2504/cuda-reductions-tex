void reduce(int * h_in, int * h_out, int len) {
    // copy data into temporary array
    int * temp = (int *) malloc(len*sizeof(int));
    memcpy(temp, h_in, len*sizeof(int));

    int stride = 1;
    // iterate over reduction steps (vertically)
    for (int step = 0; step < log2(len); step++) {

        // iterate over dataset (horizontally)
        for (int i = 0; i < len; len += 2*stride) {
            temp[i] = temp[i] + temp[i + stride];
        }
        stride *= 2;
    }

    // return the result
    *h_out = temp[0];
    free(temp);
    return;
}
