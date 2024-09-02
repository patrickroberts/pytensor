#include <tt/core/dtype.hpp>
#include <tt/core/float.hpp>
#include <tt/core/format.hpp>
#include <tt/core/int.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/tensor.hpp>
#include <tt/operators/arange.hpp>
#include <tt/operators/full.hpp>

#include <boost/mp11.hpp>
#include <magic_enum.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <format>
#include <memory>

consteval auto name_of(tt::Float32) { return "Float32"; }
consteval auto name_of(tt::Float64) { return "Float64"; }
consteval auto name_of(tt::BFloat16) { return "BFloat16"; }
consteval auto name_of(tt::UInt8) { return "UInt8"; }
consteval auto name_of(tt::Int8) { return "Int8"; }
consteval auto name_of(tt::Int16) { return "Int16"; }
consteval auto name_of(tt::Int32) { return "Int32"; }
consteval auto name_of(tt::Int64) { return "Int64"; }
consteval auto name_of(tt::Bool) { return "Bool"; }

consteval auto name_of(tt::dims<0>) { return "Scalar"; }
consteval auto name_of(tt::dims<1>) { return "Vector"; }
consteval auto name_of(tt::dims<2>) { return "Matrix"; }
consteval auto name_of(tt::dims<3>) { return "Tensor3D"; }
consteval auto name_of(tt::dims<4>) { return "Tensor4D"; }
consteval auto name_of(tt::dims<5>) { return "Tensor5D"; }
consteval auto name_of(tt::dims<6>) { return "Tensor6D"; }
consteval auto name_of(tt::dims<7>) { return "Tensor7D"; }
consteval auto name_of(tt::dims<8>) { return "Tensor8D"; }

consteval auto name_of(tt::RowMajor) { return "RowMajor"; }
consteval auto name_of(tt::Tiled) { return "Tiled"; }

PYBIND11_MODULE(tt, m) {
  namespace py = pybind11;
  namespace mp = boost::mp11;

  constexpr auto bind_enum_entries = []<class T>(T, const py::handle &handle,
                                                 const char *name) {
    using enum_type = T;
    py::enum_<enum_type> type(handle, name);

    for (const auto &[value, name] : magic_enum::enum_entries<enum_type>()) {
      type.value(name.data(), value);
    }

    if constexpr (magic_enum::is_unscoped_enum_v<enum_type>) {
      type.export_values();
    }
  };

  bind_enum_entries(tt::DType{}, m, "dtype");
  bind_enum_entries(tt::Layout{}, m, "layout");

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

  mp::mp_for_each<tensor_types>([&]<tt::tensor T>(mp::mp_identity<T>) {
    using tensor_type = T;
    using layout_type = tt::layout_type_t<tensor_type>;
    using element_type = tt::element_type_t<tensor_type>;
    using extents_type = tt::extents_type_t<tensor_type>;

    auto m_layout = m_tensor.def_submodule(name_of(layout_type{}));
    auto m_element = m_layout.def_submodule(name_of(element_type{}));
    auto c_tensor = py::class_<tensor_type>{m_element, name_of(extents_type{})};

    c_tensor.def("__repr__",
                 [](const T &tensor) { return std::format("{}", tensor); });
  });

  const auto default_dtype = std::make_shared<tt::DType>(tt::DType::Float32);

  const auto value_or_default = [=](std::optional<tt::DType> dtype) {
    return dtype ? *dtype : *default_dtype;
  };

  m.def("set_default_dtype", [=](tt::DType dtype) { *default_dtype = dtype; });

  m.def("get_default_dtype", [=] { return *default_dtype; });

  constexpr auto with_element = [](tt::DType dtype, auto callback) {
    constexpr mp::mp_size<element_types> element_types_count{};

    static_assert(element_types_count == magic_enum::enum_count<tt::DType>());

    return mp::mp_with_index<element_types_count>(
        *magic_enum::enum_index(dtype),
        [&](tt::integral_constant_like auto index) {
          using element_type = mp::mp_at_c<element_types, index>;
          return callback(element_type{});
        });
  };

  constexpr auto arange = [=](auto start, auto end, tt::Float64 step,
                              tt::DType dtype) {
    return with_element(dtype, [=]<tt::arithmetic TElement>(TElement) {
      return py::cast(tt::arange<TElement>(start, end, step));
    });
  };

  constexpr tt::Int64 default_start = 0;
  constexpr tt::Int64 default_step = 1;

  m.def(
      "arange",
      [=](tt::Int64 end, tt::DType dtype) {
        return arange(default_start, end, default_step, dtype);
      },
      py::arg("end"), py::kw_only(), py::arg("dtype") = tt::DType::Int64);

  m.def(
      "arange",
      [=](tt::Float64 end, std::optional<tt::DType> dtype) {
        return arange(default_start, end, default_step,
                      value_or_default(dtype));
      },
      py::arg("end"), py::kw_only(), py::arg("dtype") = py::none());

  m.def(
      "arange",
      [=](tt::Int64 start, tt::Int64 end, tt::Float64 step, tt::DType dtype) {
        return arange(start, end, step, dtype);
      },
      py::arg("start"), py::arg("end"), py::arg("step") = 1, py::kw_only(),
      py::arg("dtype") = tt::DType::Int64);

  m.def(
      "arange",
      [=](tt::Float64 start, tt::Float64 end, tt::Float64 step,
          std::optional<tt::DType> dtype) {
        return arange(start, end, step, value_or_default(dtype));
      },
      py::arg("start"), py::arg("end"), py::arg("step") = 1, py::kw_only(),
      py::arg("dtype") = py::none());

  constexpr auto with_extents = [](const py::args &args, auto callback) {
    constexpr mp::mp_size<extents_types> extents_types_count{};

    return mp::mp_with_index<extents_types_count>(
        args.size(), [&](tt::integral_constant_like auto rank) {
          return [&]<auto... Is>(std::index_sequence<Is...>) {
            return callback(py::cast<std::size_t>(args[Is])...);
          }(std::make_index_sequence<rank>{});
        });
  };

  constexpr auto with_element_and_extents =
      [=](tt::DType dtype, const py::args &args, auto callback) {
        return with_element(dtype, [&](tt::arithmetic auto element) {
          return with_extents(args, [&](auto... extents) {
            return callback(element, extents...);
          });
        });
      };

  const auto with_fill = [=](auto fill_value) {
    return [=](const py::args &args, std::optional<tt::DType> dtype) {
      return with_element_and_extents(
          value_or_default(dtype), args,
          [&]<tt::arithmetic TElement>(TElement, auto... extents) {
            return py::cast(
                tt::full(static_cast<TElement>(fill_value), extents...));
          });
    };
  };

  m.def("ones", with_fill(1), py::kw_only(), py::arg("dtype") = py::none());

  m.def("zeros", with_fill(0), py::kw_only(), py::arg("dtype") = py::none());
}
