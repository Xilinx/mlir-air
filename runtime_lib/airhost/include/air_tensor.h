#ifndef AIR_TENSOR_H
#define AIR_TENSOR_H

template<typename T, int N>
struct tensor_t {
  T *d;
  T *aligned;
  size_t offset;
  size_t shape[N];
  size_t stride[N];

  size_t index(size_t n, size_t channel, size_t row, size_t col) const {
    size_t channels = shape[1];
    size_t height = shape[2];
    size_t width = shape[3];
    size_t idx = n * height * width * channels + channel * height * width + row * width + col;
    if (idx >= shape[0]*shape[1]*shape[2]*shape[3]) {
      //printf("warning\n");
      return 0;
    }
    return idx;
  }

  tensor_t() {
    d = aligned = nullptr;
    offset = 0;
    for (int i=0; i<N; i++)
      shape[i] = stride[i] = 0;
  }
};

#endif