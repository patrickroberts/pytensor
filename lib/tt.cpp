#include <tt/core/dtype.hpp>
#include <tt/core/float.hpp>
#include <tt/core/format.hpp>
#include <tt/core/int.hpp>
#include <tt/core/layout.hpp>
#include <tt/core/tensor.hpp>
#include <tt/operators/arange.hpp>
#include <tt/operators/ones.hpp>
#include <tt/operators/zeros.hpp>

#include <boost/mp11.hpp>
#include <magic_enum.hpp>
#include <magic_enum_switch.hpp>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <format>
#include <memory>

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
constexpr auto name_of(tt::Tiled) { return "Tiled"; }

template <std::size_t, class T>
using dependent = T;

PYBIND11_MODULE(tt, m) {
  namespace py = pybind11;
  namespace mp = boost::mp11;

  py::enum_<tt::DType> dtype(m, "dtype");
  py::enum_<tt::Layout> layout(m, "layout");

  for (const auto &[value, name] : magic_enum::enum_entries<tt::DType>()) {
    dtype.value(name.data(), value);
  }

  for (const auto &[value, name] : magic_enum::enum_entries<tt::Layout>()) {
    layout.value(name.data(), value);
  }

  using element_types =
      mp::mp_list<tt::Float32, tt::Float64, tt::BFloat16, tt::UInt8, tt::Int8,
                  tt::Int16, tt::Int32, tt::Int64, tt::Bool>;
  using extents_types = mp::mp_list<tt::dims<0>, tt::dims<1>, tt::dims<2>,
                                    tt::dims<3>, tt::dims<4>, tt::dims<5>,
                                    tt::dims<6>, tt::dims<7>, tt::dims<8>>;
  using layout_types = mp::mp_list<tt::RowMajor, tt::Tiled>;

  auto m_tensor = m.def_submodule("Tensor");

  mp::mp_for_each<
      mp::mp_product<mp::mp_list, element_types, extents_types, layout_types>>(
      [&]<class TElement, class TExtents, class TLayout>(
          mp::mp_list<TElement, TExtents, TLayout>) {
        if constexpr (TExtents::rank() >= 2 or
                      not std::is_same_v<TLayout, tt::Tiled>) {
          using tensor_type = tt::Tensor<TElement, TExtents, TLayout>;

          py::class_<tensor_type> c_tensor{
              m_tensor.def_submodule(name_of(TLayout{}))
                  .def_submodule(name_of(TElement{})),
              name_of(TExtents{}),
          };

          c_tensor.def("__repr__", [](const tensor_type &tensor) {
            return std::format("{}", tensor);
          });
        }
      });

  const auto default_dtype = std::make_shared<tt::DType>(tt::DType::Float32);

  m.def("set_default_dtype", [=](tt::DType dtype) { *default_dtype = dtype; });

  m.def("get_default_dtype", [=] { return *default_dtype; });

  constexpr auto arange = [](tt::arithmetic auto start, tt::arithmetic auto end,
                             tt::Float64 step, tt::DType dtype) {
    return magic_enum::enum_switch(
        [=](auto dtype) {
          constexpr auto index = static_cast<std::size_t>(dtype());
          using TElement = mp::mp_at_c<element_types, index>;
          return py::cast(tt::arange<TElement>(start, end, step));
        },
        dtype);
  };

  m.def(
      "arange",
      [=](tt::Int64 start, tt::Int64 end, tt::Float64 step, tt::DType dtype) {
        return arange(start, end, step, dtype);
      },
      py::arg("start"), py::arg("end"), py::arg("step") = 1, py::kw_only(),
      py::arg("dtype") = tt::DType::Int64);

  m.def(
      "arange",
      [=](tt::Int64 end, tt::DType dtype) { return arange(0, end, 1, dtype); },
      py::arg("end"), py::kw_only(), py::arg("dtype") = tt::DType::Int64);

  m.def(
      "arange",
      [=](tt::Float64 start, tt::Float64 end, tt::Float64 step,
          std::optional<tt::DType> dtype) {
        return arange(start, end, step, dtype ? *dtype : *default_dtype);
      },
      py::arg("start"), py::arg("end"), py::arg("step") = 1, py::kw_only(),
      py::arg("dtype") = py::none());

  m.def(
      "arange",
      [=](tt::Float64 end, std::optional<tt::DType> dtype) {
        return arange(0, end, 1, dtype ? *dtype : *default_dtype);
      },
      py::arg("end"), py::kw_only(), py::arg("dtype") = py::none());

  m.def(
      "ones",
      [=](const py::args &args, std::optional<tt::DType> dtype) {
        return magic_enum::enum_switch(
            [&](auto dtype) {
              constexpr auto index = static_cast<std::size_t>(dtype());
              using TElement = mp::mp_at_c<element_types, index>;

              return mp::mp_with_index<mp::mp_size<extents_types>::value>(
                  args.size(), [&](auto rank) {
                    return [&]<auto... Is>(std::index_sequence<Is...>) {
                      return py::cast(tt::ones<TElement>(
                          py::cast<std::size_t>(args[Is])...));
                    }(std::make_index_sequence<rank>{});
                  });
            },
            dtype ? *dtype : *default_dtype);
      },
      py::kw_only(), py::arg("dtype") = py::none());

  m.def(
      "zeros",
      [=](const py::args &args, std::optional<tt::DType> dtype) {
        return magic_enum::enum_switch(
            [&](auto dtype) {
              constexpr auto index = static_cast<std::size_t>(dtype());
              using TElement = mp::mp_at_c<element_types, index>;

              return mp::mp_with_index<mp::mp_size<extents_types>::value>(
                  args.size(), [&](auto rank) {
                    return [&]<auto... Is>(std::index_sequence<Is...>) {
                      return py::cast(tt::zeros<TElement>(
                          py::cast<std::size_t>(args[Is])...));
                    }(std::make_index_sequence<rank>{});
                  });
            },
            dtype ? *dtype : *default_dtype);
      },
      py::kw_only(), py::arg("dtype") = py::none());
}
