### Pytorch

[pytorch/DistanceKernel.cu at main · pytorch/pytorch (github.com)](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistanceKernel.cu#L70)



### Paddle

[【PaddlePaddle Hackathon 4 No.39】为 Paddle 优化 p_norm_grad op 在 GPU 上的计算性能 by zbt78 · Pull Request #54156 · PaddlePaddle/Paddle (github.com)](https://github.com/PaddlePaddle/Paddle/pull/54156)

### 范数梯度计算

当计算一个张量的 P 范数的梯度时，对于张量中的每个元素，其梯度值需要根据该元素的值、P 范数值和该元素在输出张量中的梯度值来计算。具体来说，对于输入张量中的第 $i$ 个元素 $x_i$，其在输出张量中的梯度值为 $dy_i$，则该元素的梯度值为：
$$
\frac{\partial}{\partial x_i} \|x\|_p = \mathrm{sgn}(x_i) \cdot |x_i|^{p-1} \cdot dy_i \cdot (y+\epsilon)^{-p}
$$

其中，$\mathrm{sgn}(x_i)$ 表示 $x_i$ 的符号，$|x_i|$ 表示 $x_i$ 的绝对值，$p$ 表示 P 范数的指数，$y$ 表示 P 范数的值，$\epsilon$ 表示一个小常数，避免除数为 $0$。

因此，在该函数中，最终返回值的逻辑为将上式中的各项相乘，即：

$$
\mathrm{sgn}(x_i) \cdot |x_i|^{p-1} \cdot dy_i \cdot (y+\epsilon)^{-p}
$$

这就是该函数最终返回的浮点数类型的值，表示对应元素的梯度值。

### 代码

```c++
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/p_norm_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
__device__ __forceinline__ int inline_sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
__device__ __forceinline__ T inline_pow(T base, T exponent) {
  return static_cast<T>(
      pow(static_cast<float>(base), static_cast<float>(exponent)));
}

template <>
__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

template <typename T>
__device__ __forceinline__ T inline_abs(T x) {
  return static_cast<T>(abs(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ double inline_abs(double x) {
  return abs(x);
}

template <typename T>
struct PNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;
  const T porder_;

 public:
  HOSTDEVICE inline PNormGradScalarDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon,
                                                     const T porder)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon),
        porder_(porder - static_cast<T>(1.)) {}

  HOSTDEVICE inline T operator()(const T x) const {
    const T scalar =
        dy_[0] * inline_pow<T>(y_[0] + epsilon_, static_cast<T>(-1) * porder_);
    return static_cast<T>(static_cast<T>(inline_sgn<T>(x)) *
                          inline_pow<T>(inline_abs<T>(x), porder_) * scalar);
  }
};

template <typename T>
struct InfinityNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;

 public:
  HOSTDEVICE inline InfinityNormGradScalarDirectCUDAFunctor(const T* y,
                                                            const T* dy)
      : y_(y), dy_(dy) {}

  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(dy_[0] * static_cast<T>(inline_sgn<T>(x)) *
                          static_cast<T>((inline_abs<T>(x) == y_[0])));
  }
};

template <typename T>
struct OneNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;

 public:
  HOSTDEVICE inline OneNormGradScalarDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon) {}

  HOSTDEVICE inline T operator()(const T x) const {
    // return grad * sign(diff);
    return static_cast<T>(dy_[0] * static_cast<T>(inline_sgn<T>(x)));
  }
};

template <typename T>
struct LtTwoNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;
  const T porder_;

 public:
  HOSTDEVICE inline LtTwoNormGradScalarDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon,
                                                     const T porder)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon),
        porder_(porder - static_cast<T>(1.)) {}

  HOSTDEVICE inline T operator()(const T x) const {
    // sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1)
    const T tmp = inline_sgn<T>(y_[0]) * inline_pow<T>(inline_abs(y_[0]), porder_) * dy_[0] / inline_pow<T>(y_[0], porder_);
    // return (dist == 0.0 || (diff == 0.0 && p < 1)) ? 0 : (sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1));
    return (inline_abs<T>(y_[0]) < static_cast<T>(1e-5) || (inline_abs<T>(y_[0]) < static_cast<T>(1e-5) && porder_ < 1)) ? static_cast<T>(0) : tmp;
  }
};

template <typename T>
struct TwoNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;

 public:
  HOSTDEVICE inline TwoNormGradScalarDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon) {}

  HOSTDEVICE inline T operator()(const T x) const {
    // return dist == 0.0 ? 0 : grad * diff / dist;
    return static_cast<T>(inline_abs<T>(y_[0]) < static_cast<T>(1e-5) ? static_cast<T>(0) : dy_[0] * x / y_[0]);
    // return static_cast<T>(y_[0] == static_cast<T>(0) ? static_cast<T>(0) : dy_[0] * x / y_[0]);

  }
};

template <typename T>
struct InfinityNormGradTensorDirectCUDAFunctor {
  HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
    return static_cast<T>(dy * static_cast<T>(inline_sgn<T>(x)) *
                          static_cast<T>(inline_abs<T>(x) == y));
  }
};

template <typename T>
struct OneNormGradTensorDirectCUDAFunctor {
  private:
    const T epsilon_;
  public:
    HOSTDEVICE inline OneNormGradTensorDirectCUDAFunctor(const T epsilon)
      : epsilon_(epsilon) {}
    HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
      return static_cast<T>(dy * static_cast<T>(inline_sgn<T>(x)));
    }
};

template <typename T>
struct LtTwoNormGradTensorDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;
  const T porder_;

 public:
  HOSTDEVICE inline LtTwoNormGradTensorDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon,
                                                     const T porder)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon),
        porder_(porder - static_cast<T>(1.)) {}

  HOSTDEVICE inline T operator()(const T x) const {
    // sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1)
    const T tmp = inline_sgn<T>(y_) * inline_pow<T>(inline_abs(y_), porder_) * dy_ / inline_pow<T>(y_, porder_);
    // return (dist == 0.0 || (diff == 0.0 && p < 1)) ? 0 : (sign(diff) * std::pow(std::abs(diff), p - 1) * grad / std::pow(dist, p - 1));
    return (inline_abs<T>(y_[0]) < static_cast<T>(1e-5) || (inline_abs<T>(y_) < static_cast<T>(1e-5) && porder_ < 1)) ? static_cast<T>(0) : tmp;
  }
};

template <typename T>
struct TwoNormGradTensorDirectCUDAFunctor {
  private:
    const T epsilon_;
  public:
    HOSTDEVICE inline TwoNormGradTensorDirectCUDAFunctor(const T epsilon)
      : epsilon_(epsilon) {}

    HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
      // return dist == 0.0 ? 0 : grad * diff / dist;
      return static_cast<T>(inline_abs<T>(y) < static_cast<T>(1e-5) ? static_cast<T>(0) : dy * x / y);
    }
};

template <typename T>
struct PNormGradTensorDirectCUDAFunctor {
 private:
  const T epsilon_;
  const T porder_;

 public:
  HOSTDEVICE inline PNormGradTensorDirectCUDAFunctor(const T epsilon,
                                                     const T porder)
      : epsilon_(epsilon), porder_(porder - static_cast<T>(1.)) {}

  HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
    return static_cast<T>(
        static_cast<T>(inline_sgn<T>(x)) *
        inline_pow<T>(inline_abs<T>(x), porder_) * dy *
        inline_pow<T>(y + epsilon_, static_cast<T>(-1.0) * porder_));
  }
};

template <typename T, typename Context>
void PNormGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector,
                     DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  bool reduce_all = (out.numel() == 1);
  if (porder == 0) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, x_grad, static_cast<T>(0));
  } else {
    std::vector<DenseTensor*> outputs = {x_grad};
    if (reduce_all) {
      std::vector<const DenseTensor*> inputs = {&x};

      const T* out_ptr = out.data<T>();
      const T* out_grad_ptr = out_grad.data<T>();
      if (porder == INFINITY || porder == -INFINITY) {
        auto functor =
            InfinityNormGradScalarDirectCUDAFunctor<T>(out_ptr, out_grad_ptr);
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 1) {
        auto functor =
            OneNormGradScalarDirectCUDAFunctor<T>(out_ptr,
                                                out_grad_ptr,
                                                static_cast<T>(epsilon));
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder < 2) {
        // auto functor =
        //     LtTwoNormGradScalarDirectCUDAFunctor<T>(out_ptr,
        //                                         out_grad_ptr,
        //                                         static_cast<T>(epsilon));
        // funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 2) {
        auto functor =
            TwoNormGradScalarDirectCUDAFunctor<T>(out_ptr,
                                                out_grad_ptr,
                                                static_cast<T>(epsilon));
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else {
        auto functor =
            PNormGradScalarDirectCUDAFunctor<T>(out_ptr,
                                                out_grad_ptr,
                                                static_cast<T>(epsilon),
                                                static_cast<T>(porder));
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      }
    } else {
      if (axis < 0) axis += x.dims().size();
      std::vector<int> shape;
      for (int i = 0; i < x.dims().size(); i++) {
        if (i < axis) {
          shape.push_back(out.dims()[i]);
        } else if (i == axis) {
          shape.push_back(1);
        } else {
          shape.push_back(out.dims()[i - 1]);
        }
      }
      DenseTensor out_copy(out);
      DenseTensor out_grad_copy(out_grad);
      if (!keepdim) {
        DDim dims = phi::make_ddim(shape);
        out_copy.Resize(dims);
        out_grad_copy.Resize(dims);
      }
      std::vector<const DenseTensor*> inputs = {&x, &out_copy, &out_grad_copy};
      if (porder == INFINITY || porder == -INFINITY) {
        auto functor = InfinityNormGradTensorDirectCUDAFunctor<T>();
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 1) {
        auto functor = OneNormGradTensorDirectCUDAFunctor<T>(
            static_cast<T>(epsilon));
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder < 2) {
        // auto functor = LtTwoNormGradTensorDirectCUDAFunctor<T>(
        //     static_cast<T>(epsilon), static_cast<T>(porder));
        // funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 2) {
        auto functor = TwoNormGradTensorDirectCUDAFunctor<T>(
            static_cast<T>(epsilon));
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else {
        auto functor = PNormGradTensorDirectCUDAFunctor<T>(
            static_cast<T>(epsilon), static_cast<T>(porder));
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      }
    }
  }
}


}  // namespace phi
PD_REGISTER_KERNEL(p_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::PNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
```





pytorch改进

```c++
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/p_norm_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
__device__ __forceinline__ int inline_sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
__device__ __forceinline__ T inline_pow(T base, T exponent) {
  return static_cast<T>(
      pow(static_cast<float>(base), static_cast<float>(exponent)));
}

template <>
__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

template <typename T>
__device__ __forceinline__ T inline_abs(T x) {
  return static_cast<T>(abs(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ double inline_abs(double x) {
  return abs(x);
}

template <typename T>
struct PNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;
  const T porder_;

 public:
  HOSTDEVICE inline PNormGradScalarDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon,
                                                     const T porder)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon),
        porder_(porder - static_cast<T>(1.)) {}

  HOSTDEVICE inline T operator()(const T x) const {
    const T scalar =
        dy_[0] * inline_pow<T>(y_[0] + epsilon_, static_cast<T>(-1) * porder_);
    return static_cast<T>(static_cast<T>(inline_sgn<T>(x)) *
                          inline_pow<T>(inline_abs<T>(x), porder_) * scalar);
  }
};

template <typename T>
struct OneNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;

 public:
  HOSTDEVICE inline OneNormGradScalarDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon) {}

  HOSTDEVICE inline T operator()(const T x) const {
    // return grad * sign(diff);
    return static_cast<T>(dy_[0] * static_cast<T>(inline_sgn<T>(x)));
  }
};

template <typename T>
struct TwoNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;
  const T epsilon_;

 public:
  HOSTDEVICE inline TwoNormGradScalarDirectCUDAFunctor(const T* y,
                                                     const T* dy,
                                                     const T epsilon)
      : y_(y),
        dy_(dy),
        epsilon_(epsilon) {}

  HOSTDEVICE inline T operator()(const T x) const {
    // return dist == 0.0 ? 0 : grad * diff / dist;
    return static_cast<T>(y_[0] == 0.0 ? 0 : dy_[0] * x / y_[0]);
  }
};

template <typename T>
struct InfinityNormGradScalarDirectCUDAFunctor {
 private:
  const T* y_;
  const T* dy_;

 public:
  HOSTDEVICE inline InfinityNormGradScalarDirectCUDAFunctor(const T* y,
                                                            const T* dy)
      : y_(y), dy_(dy) {}
  // return grad * sign(diff) * (std::abs(diff) == dist);
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(dy_[0] * static_cast<T>(inline_sgn<T>(x)) *
                          static_cast<T>((inline_abs<T>(x) == y_[0])));
  }
};

template <typename T>
struct OneNormGradTensorDirectCUDAFunctor {
  private:
    const T epsilon_;
  public:
    HOSTDEVICE inline OneNormGradTensorDirectCUDAFunctor(const T epsilon)
      : epsilon_(epsilon) {}
    HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
      return static_cast<T>(dy * static_cast<T>(inline_sgn<T>(x)));
    }
};

template <typename T>
struct TwoNormGradTensorDirectCUDAFunctor {
  private:
    const T epsilon_;
  public:
    HOSTDEVICE inline TwoNormGradTensorDirectCUDAFunctor(const T epsilon)
      : epsilon_(epsilon) {}

    HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
      // return dist == 0.0 ? 0 : grad * diff / dist;
      return static_cast<T>(y == 0.0 ? 0 : dy * x / y);
    }
};

template <typename T>
struct InfinityNormGradTensorDirectCUDAFunctor {
  HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
    return static_cast<T>(dy * static_cast<T>(inline_sgn<T>(x)) *
                          static_cast<T>(inline_abs<T>(x) == y));
  }
};

template <typename T>
struct PNormGradTensorDirectCUDAFunctor {
 private:
  const T epsilon_;
  const T porder_;

 public:
  HOSTDEVICE inline PNormGradTensorDirectCUDAFunctor(const T epsilon,
                                                     const T porder)
      : epsilon_(epsilon), porder_(porder - static_cast<T>(1.)) {}

  HOSTDEVICE inline T operator()(const T x, const T y, const T dy) const {
    return static_cast<T>(
        static_cast<T>(inline_sgn<T>(x)) *
        inline_pow<T>(inline_abs<T>(x), porder_) * dy *
        inline_pow<T>(y + epsilon_, static_cast<T>(-1.0) * porder_));
  }
};

template <typename T, typename Context>
void PNormGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector,
                     DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  bool reduce_all = (out.numel() == 1);
  if (porder == 0) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, x_grad, static_cast<T>(0));
  } else {
    std::vector<DenseTensor*> outputs = {x_grad};
    if (reduce_all) {
      std::vector<const DenseTensor*> inputs = {&x};

      const T* out_ptr = out.data<T>();
      const T* out_grad_ptr = out_grad.data<T>();
      if (porder == INFINITY || porder == -INFINITY) {
        auto functor =
            InfinityNormGradScalarDirectCUDAFunctor<T>(out_ptr, out_grad_ptr);
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 1) {
        auto functor =
            OneNormGradScalarDirectCUDAFunctor<T>(out_ptr, out_grad_ptr, static_cast<T>(epsilon));
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 2) {
        auto functor =
            TwoNormGradScalarDirectCUDAFunctor<T>(out_ptr, out_grad_ptr, static_cast<T>(epsilon));
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else {
        auto functor =
            PNormGradScalarDirectCUDAFunctor<T>(out_ptr,
                                                out_grad_ptr,
                                                static_cast<T>(epsilon),
                                                static_cast<T>(porder));
        funcs::ElementwiseKernel<T>(dev_ctx, inputs, &outputs, functor);
      }
    } else {
      if (axis < 0) axis += x.dims().size();
      std::vector<int> shape;
      for (int i = 0; i < x.dims().size(); i++) {
        if (i < axis) {
          shape.push_back(out.dims()[i]);
        } else if (i == axis) {
          shape.push_back(1);
        } else {
          shape.push_back(out.dims()[i - 1]);
        }
      }
      DenseTensor out_copy(out);
      DenseTensor out_grad_copy(out_grad);
      if (!keepdim) {
        DDim dims = phi::make_ddim(shape);
        out_copy.Resize(dims);
        out_grad_copy.Resize(dims);
      }
      std::vector<const DenseTensor*> inputs = {&x, &out_copy, &out_grad_copy};
      if (porder == INFINITY || porder == -INFINITY) {
        auto functor = InfinityNormGradTensorDirectCUDAFunctor<T>();
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 1) {
        auto functor = OneNormGradTensorDirectCUDAFunctor<T>(
            static_cast<T>(epsilon));
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else if(porder == 2) {
        auto functor = TwoNormGradTensorDirectCUDAFunctor<T>(
            static_cast<T>(epsilon));
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      } else {
        auto functor = PNormGradTensorDirectCUDAFunctor<T>(
            static_cast<T>(epsilon), static_cast<T>(porder));
        funcs::BroadcastKernel<T>(dev_ctx, inputs, &outputs, functor);
      }
    }
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(p_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::PNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
```

