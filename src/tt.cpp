#include <tt/core/dtype.hpp>
#include <tt/core/float.hpp>
#include <tt/core/format.hpp>
#include <tt/core/int.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/tensor.hpp>
#include <tt/operators/arange.hpp>
#include <tt/operators/empty.hpp>
#include <tt/operators/eye.hpp>
#include <tt/operators/full.hpp>

#include <boost/mp11.hpp>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <memory>
#include <stdexcept>

constexpr auto name_of(tt::Float32) { return "Float32"; }
constexpr auto name_of(tt::Float64) { return "Float64"; }
constexpr auto name_of(tt::BFloat16) { return "BFloat16"; }
constexpr auto name_of(tt::UInt8) { return "UInt8"; }
constexpr auto name_of(tt::Int8) { return "Int8"; }
constexpr auto name_of(tt::Int16) { return "Int16"; }
constexpr auto name_of(tt::Int32) { return "Int32"; }
constexpr auto name_of(tt::Int64) { return "Int64"; }
constexpr auto name_of(tt::Bool) { return "Bool"; }

constexpr auto name_of(tt::dims<0>) { return "Scalar"; }
constexpr auto name_of(tt::dims<1>) { return "Vector"; }
constexpr auto name_of(tt::dims<2>) { return "Matrix"; }
constexpr auto name_of(tt::dims<3>) { return "Tensor3D"; }
constexpr auto name_of(tt::dims<4>) { return "Tensor4D"; }
constexpr auto name_of(tt::dims<5>) { return "Tensor5D"; }
constexpr auto name_of(tt::dims<6>) { return "Tensor6D"; }
constexpr auto name_of(tt::dims<7>) { return "Tensor7D"; }
constexpr auto name_of(tt::dims<8>) { return "Tensor8D"; }

constexpr auto name_of(tt::RowMajor) { return "RowMajor"; }
constexpr auto name_of(tt::Strided) { return "Strided"; }
constexpr auto name_of(tt::Tiled) { return "Tiled"; }

namespace py = nanobind;
namespace mp = boost::mp11;

template <class TEnum, class = std::enable_if_t<std::is_enum_v<TEnum>>>
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
constexpr auto apply_extents(const py::args &extents, TCallback callback,
                             std::index_sequence<Is...>) {
  return callback(py::cast<std::size_t>(extents[Is])...);
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
  using non_tiled_tensor_types =
      mp::mp_product<tt::Tensor, element_types, extents_types,
                     mp::mp_list<tt::RowMajor, tt::Strided>>;
  using tiled_tensor_types = mp::mp_product<
      tt::Tensor, element_types,
      mp::mp_list<tt::dims<2>, tt::dims<3>, tt::dims<4>, tt::dims<5>,
                  tt::dims<6>, tt::dims<7>, tt::dims<8>>,
      mp::mp_list<tt::Tiled>>;
  using tensor_types =
      mp::mp_transform<mp::mp_identity, mp::mp_append<non_tiled_tensor_types,
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

  constexpr auto visit_enum = [](auto value, auto callback) {
    using enum_type = decltype(value);
    static_assert(std::is_enum_v<enum_type>);

    return mp::mp_with_index<magic_enum::enum_count<enum_type>()>(
        *magic_enum::enum_index(value), [&](auto index) {
          constexpr auto value = magic_enum::enum_value<enum_type, index>();
          return callback(tt::constant<value>{});
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

          return visit_enum(arg, [&](auto dtype) {
            return py::cast(tt::arange<dtype()>(args...));
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

  constexpr auto visit_extents = [](const py::args &extents, auto callback) {
    using extents_size = mp::mp_size<extents_types>;

    if (extents.size() >= extents_size::value) {
      throw std::range_error(
          fmt::format("len(extents) {} not supported; must be less than {}",
                      extents.size(), extents_size::value));
    }

    for (std::size_t index = 0; index < extents.size(); ++index) {
      if (not py::isinstance<std::size_t>(extents[index])) {
        throw py::type_error(py::str("expected extents[{}] to be {}; got {}")
                                 .format(index, py::inst_name(py::cast(index)),
                                         py::repr(extents[index]))
                                 .c_str());
      }
    }

    return mp::mp_with_index<extents_size>(extents.size(), [&](auto rank) {
      return apply_extents(extents, callback, std::make_index_sequence<rank>{});
    });
  };

  constexpr auto visit_dtype_and_extents =
      [=](tt::dtype dtype, const py::args &extents, auto callback) {
        return visit_enum(dtype, [&](auto dtype) {
          return visit_extents(extents, [&](auto... extents) {
            return callback(dtype, extents...);
          });
        });
      };

  const auto bind_with_fill = [=](auto fill_value) {
    return [=](const py::args &extents, std::optional<tt::dtype> dtype) {
      return visit_dtype_and_extents(
          value_or_default(dtype), extents, [&](auto dtype, auto... extents) {
            return py::cast(tt::full<dtype()>(fill_value, extents...));
          });
    };
  };

  m.def(
      "full",
      [=](tt::Float64 fill_value, const py::args &extents,
          std::optional<tt::dtype> dtype) {
        return visit_dtype_and_extents(
            value_or_default(dtype), extents, [&](auto dtype, auto... extents) {
              return py::cast(tt::full<dtype()>(fill_value, extents...));
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
      [=](const py::args &extents, std::optional<tt::dtype> dtype) {
        return visit_dtype_and_extents(
            value_or_default(dtype), extents, [](auto dtype, auto... extents) {
              return py::cast(tt::empty<dtype()>(extents...));
            });
      },
      py::arg("extents"), py::kw_only(), py::arg("dtype") = py::none());

  m.def(
      "eye",
      [=](std::size_t extent, std::optional<tt::dtype> dtype) {
        return visit_enum(value_or_default(dtype), [&](auto dtype) {
          return py::cast(tt::eye<dtype()>(extent));
        });
      },
      py::arg("extent"), py::kw_only(), py::arg("dtype") = py::none());

  m.def(
      "eye",
      [=](std::size_t rows, std::size_t cols, std::optional<tt::dtype> dtype) {
        return visit_enum(value_or_default(dtype), [&](auto dtype) {
          return py::cast(tt::eye<dtype()>(rows, cols));
        });
      },
      py::arg("rows"), py::arg("cols"), py::kw_only(),
      py::arg("dtype") = py::none());
}
