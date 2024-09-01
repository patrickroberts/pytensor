#include <tt/core/bfloat16.hpp>
#include <tt/core/concepts.hpp>

#include <complex>

namespace tt::inline core {

using complex64 = std::complex<tt::float32>;
using complex128 = std::complex<tt::float64>;

} // namespace tt::inline core

template <class T>
inline constexpr bool tt::is_arithmetic_v<std::complex<T>> = true;
