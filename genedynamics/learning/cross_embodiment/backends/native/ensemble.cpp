// ABI v1: caller owns contiguous float32 weights/input/output, no retained pointers.
#include <cmath>
#include <cstdint>
#include <vector>
#include <exception>
extern "C" int cross_embodiment_abi_version() { return 1; }
extern "C" int cross_embodiment_mlp(
    const float* weights, const float* biases, const int32_t* dims,
    int32_t layers, int32_t ensemble, int32_t batch,
    const float* inputs, float* outputs) noexcept {
  if (!weights || !biases || !dims || !inputs || !outputs || layers < 1 ||
      layers > 64 || ensemble < 1 || batch < 1) return 1;
  for (int i=0;i<=layers;++i) if(dims[i]<1 || dims[i]>65536) return 2;
  try {
    for(int e=0;e<ensemble;++e) {
      for(int b=0;b<batch;++b) {
        std::vector<float> x(inputs+b*dims[0],inputs+(b+1)*dims[0]);
        int64_t wbase=0,bbase=0;
        for(int l=0;l<layers;++l) {
          int n=dims[l],m=dims[l+1];
          std::vector<float> y(m);
          for(int j=0;j<m;++j) {
            float v=biases[bbase+e*m+j];
            for(int i=0;i<n;++i) v += x[i]*weights[wbase+(e*n+i)*m+j];
            y[j]=(l+1<layers)?std::tanh(v):v;
          }
          wbase+=static_cast<int64_t>(ensemble)*n*m;
          bbase+=static_cast<int64_t>(ensemble)*m;
          x.swap(y);
        }
        for(int j=0;j<dims[layers];++j) {
          if(!std::isfinite(x[j])) return 3;
          outputs[(e*batch+b)*dims[layers]+j]=x[j];
        }
      }
    }
  } catch(const std::exception&) {return 4;} catch(...) {return 5;}
  return 0;
}
