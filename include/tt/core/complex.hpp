#include <tt/core/concepts.hpp>
#include <tt/core/float.hpp>

#include <complex>

namespace tt {
inline namespace core {

using Complex64 = std::complex<tt::Float32>;
using Complex128 = std::complex<tt::Float64>;

} // namespace core
} // namespace tt

template <class T>
inline constexpr bool tt::is_arithmetic_v<std::complex<T>> =
    tt::is_arithmetic_v<T>;
