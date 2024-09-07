#include <tt/core/dtype.hpp>
#include <tt/core/float.hpp>
#include <tt/core/format.hpp>
#include <tt/core/int.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/tensor.hpp>
#include <tt/operators/arange.hpp>
#include <tt/operators/empty.hpp>
#include <tt/operators/full.hpp>

#include <boost/mp11.hpp>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <memory>

TT_CONSTEVAL auto name_of(tt::Float32) { return "Float32"; }
TT_CONSTEVAL auto name_of(tt::Float64) { return "Float64"; }
TT_CONSTEVAL auto name_of(tt::BFloat16) { return "BFloat16"; }
TT_CONSTEVAL auto name_of(tt::UInt8) { return "UInt8"; }
TT_CONSTEVAL auto name_of(tt::Int8) { return "Int8"; }
TT_CONSTEVAL auto name_of(tt::Int16) { return "Int16"; }
TT_CONSTEVAL auto name_of(tt::Int32) { return "Int32"; }
TT_CONSTEVAL auto name_of(tt::Int64) { return "Int64"; }
TT_CONSTEVAL auto name_of(tt::Bool) { return "Bool"; }

TT_CONSTEVAL auto name_of(tt::dims<0>) { return "Scalar"; }
TT_CONSTEVAL auto name_of(tt::dims<1>) { return "Vector"; }
TT_CONSTEVAL auto name_of(tt::dims<2>) { return "Matrix"; }
TT_CONSTEVAL auto name_of(tt::dims<3>) { return "Tensor3D"; }
TT_CONSTEVAL auto name_of(tt::dims<4>) { return "Tensor4D"; }
TT_CONSTEVAL auto name_of(tt::dims<5>) { return "Tensor5D"; }
TT_CONSTEVAL auto name_of(tt::dims<6>) { return "Tensor6D"; }
TT_CONSTEVAL auto name_of(tt::dims<7>) { return "Tensor7D"; }
TT_CONSTEVAL auto name_of(tt::dims<8>) { return "Tensor8D"; }

TT_CONSTEVAL auto name_of(tt::RowMajor) { return "RowMajor"; }
TT_CONSTEVAL auto name_of(tt::Tiled) { return "Tiled"; }

namespace py = nanobind;
namespace mp = boost::mp11;

template <class TEnum, class = TT_REQUIRES(std::is_enum_v<TEnum>)>
constexpr auto bind_enum(const py::handle &handle) -> void {
  constexpr auto type_name = magic_enum::enum_type_name<TEnum>();

  py::enum_<TEnum> type(handle, type_name.data());

  for (const auto &[value, entry_name] : magic_enum::enum_entries<TEnum>()) {
    type.value(entry_name.data(), value);
  }

  if constexpr (magic_enum::is_unscoped_enum_v<TEnum>) {
    type.export_values();
  }
}

template <class TCallback, std::size_t... Is>
constexpr auto apply_extents(const py::args &args, TCallback callback,
                             std::index_sequence<Is...>) {
  return callback(py::cast<std::size_t>(args[Is])...);
}

NB_MODULE(_tt, m) {
  bind_enum<tt::dtype>(m);
  bind_enum<tt::layout>(m);

  using element_types =
      mp::mp_list<tt::Float32, tt::Float64, tt::BFloat16, tt::UInt8, tt::Int8,
                  tt::Int16, tt::Int32, tt::Int64, tt::Bool>;
  using extents_types = mp::mp_list<tt::dims<0>, tt::dims<1>, tt::dims<2>,
                                    tt::dims<3>, tt::dims<4>, tt::dims<5>,
                                    tt::dims<6>, tt::dims<7>, tt::dims<8>>;
  using row_major_tensor_types =
      mp::mp_product<tt::Tensor, element_types, extents_types,
                     mp::mp_list<tt::RowMajor>>;
  using tiled_tensor_types = mp::mp_product<
      tt::Tensor, element_types,
      mp::mp_list<tt::dims<2>, tt::dims<3>, tt::dims<4>, tt::dims<5>,
                  tt::dims<6>, tt::dims<7>, tt::dims<8>>,
      mp::mp_list<tt::Tiled>>;
  using tensor_types =
      mp::mp_transform<mp::mp_identity, mp::mp_append<row_major_tensor_types,
                                                      tiled_tensor_types>>;

  auto m_tensor = m.def_submodule("Tensor");

  mp::mp_for_each<tensor_types>([&](auto identity) {
    using tensor_type = typename decltype(identity)::type;
    using layout_type = tt::layout_type_t<tensor_type>;
    using element_type = tt::element_type_t<tensor_type>;
    using extents_type = tt::extents_type_t<tensor_type>;

    auto m_layout = m_tensor.def_submodule(name_of(layout_type{}));
    auto m_element = m_layout.def_submodule(name_of(element_type{}));
    auto c_tensor = py::class_<tensor_type>{m_element, name_of(extents_type{})};

    c_tensor.def("__repr__", [](const tensor_type &tensor) {
      return fmt::format("{}", tensor);
    });
  });

  const auto default_dtype = std::make_shared<tt::dtype>(tt::dtype::Float32);

  const auto value_or_default = [=](std::optional<tt::dtype> dtype) {
    return dtype ? *dtype : *default_dtype;
  };

  m.def("set_default_dtype", [=](tt::dtype dtype) { *default_dtype = dtype; });

  m.def("get_default_dtype", [=] { return *default_dtype; });

  constexpr auto invoke_with_element = [](tt::dtype dtype, auto callback) {
    constexpr mp::mp_size<element_types> element_types_count{};

    static_assert(element_types_count == magic_enum::enum_count<tt::dtype>());

    return mp::mp_with_index<element_types_count>(
        *magic_enum::enum_index(dtype), [&](auto index) {
          using element_type = mp::mp_at_c<element_types, index>;
          return callback(element_type{});
        });
  };

  using Number = std::variant<tt::Int64, tt::Float64>;

  const auto arange = [=](Number start, Number end, Number step,
                          std::optional<tt::dtype> dtype) {
    return std::visit(
        [&](auto... args) {
          const auto arg = [&] {
            if constexpr ((... and std::is_integral_v<decltype(args)>)) {
              return dtype.value_or(tt::dtype::Int64);
            } else {
              return value_or_default(dtype);
            }
          }();

          return invoke_with_element(arg, [&](auto element) {
            using element_type = decltype(element);
            return py::cast(tt::arange<element_type>(args...));
          });
        },
        start, end, step);
  };

  constexpr tt::Int64 default_start = 0;
  constexpr tt::Int64 default_step = 1;

  m.def(
      "arange",
      [=](Number end, Number start, Number step,
          std::optional<tt::dtype> dtype) {
        return arange(start, end, step, dtype);
      },
      py::arg("end"), py::kw_only(), py::arg("start") = default_start,
      py::arg("step") = default_step, py::arg("dtype") = py::none());

  m.def("arange", arange, py::arg("start"), py::arg("end"),
        py::arg("step") = default_step, py::kw_only(),
        py::arg("dtype") = py::none());

  constexpr auto invoke_with_extents = [](const py::args &args, auto callback) {
    constexpr mp::mp_size<extents_types> extents_types_count{};

    return mp::mp_with_index<extents_types_count>(args.size(), [&](auto rank) {
      return apply_extents(args, callback, std::make_index_sequence<rank>{});
    });
  };

  constexpr auto invoke_with_element_and_extents =
      [=](tt::dtype dtype, const py::args &args, auto callback) {
        return invoke_with_element(dtype, [&](auto element) {
          return invoke_with_extents(args, [&](auto... extents) {
            return callback(element, extents...);
          });
        });
      };

  const auto bind_with_fill = [=](auto fill_value) {
    return [=](const py::args &args, std::optional<tt::dtype> dtype) {
      return invoke_with_element_and_extents(
          value_or_default(dtype), args, [&](auto element, auto... extents) {
            using element_type = decltype(element);
            return py::cast(
                tt::full(static_cast<element_type>(fill_value), extents...));
          });
    };
  };

  m.def(
      "fill",
      [=](tt::Float64 fill_value, const py::args &args,
          std::optional<tt::dtype> dtype) {
        return invoke_with_element_and_extents(
            value_or_default(dtype), args, [&](auto element, auto... extents) {
              using element_type = decltype(element);
              return py::cast(
                  tt::full(static_cast<element_type>(fill_value), extents...));
            });
      },
      py::arg("fill_value"), py::arg("extents"), py::kw_only(),
      py::arg("dtype") = py::none());

  m.def("ones", bind_with_fill(1), py::arg("extents"), py::kw_only(),
        py::arg("dtype") = py::none());

  m.def("zeros", bind_with_fill(0), py::arg("extents"), py::kw_only(),
        py::arg("dtype") = py::none());

  m.def(
      "empty",
      [=](const py::args &args, std::optional<tt::dtype> dtype) {
        return invoke_with_element_and_extents(
            value_or_default(dtype), args, [](auto element, auto... extents) {
              using element_type = decltype(element);
              return py::cast(tt::empty<element_type>(extents...));
            });
      },
      py::arg("extents"), py::kw_only(), py::arg("dtype") = py::none());
}
