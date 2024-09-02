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
#include <pybind11/pybind11.h>
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

  constexpr auto bind_enum = []<class T>(std::in_place_type_t<T>,
                                         const py::handle &handle,
                                         const char *name) {
    using enum_type = T;

    py::enum_<enum_type> type(handle, name);

    for (const auto &[value, name] : magic_enum::enum_entries<enum_type>()) {
      type.value(name.data(), value);
    }
  };

  bind_enum(std::in_place_type<tt::DType>, m, "dtype");
  bind_enum(std::in_place_type<tt::Layout>, m, "layout");

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

  mp::mp_for_each<tensor_types>([&]<class T>(mp::mp_identity<T>) {
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

  m.def("set_default_dtype", [=](tt::DType dtype) { *default_dtype = dtype; });

  m.def("get_default_dtype", [=] { return *default_dtype; });

  constexpr auto with_element = [](tt::DType dtype, auto callback) {
    return magic_enum::enum_switch(
        [&](auto dtype) {
          constexpr auto index = static_cast<std::size_t>(dtype());
          using element_type = mp::mp_at_c<element_types, index>;
          return callback(element_type{});
        },
        dtype);
  };

  constexpr auto arange = [=](auto start, auto end, tt::Float64 step,
                              tt::DType dtype) {
    return with_element(dtype, [=]<class TElement>(TElement) {
      return py::cast(tt::arange<TElement>(start, end, step));
    });
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

  constexpr auto with_extents = [=](const py::args &args, auto callback) {
    return mp::mp_with_index<mp::mp_size<element_types>::value>(
        args.size(), [&](auto rank) {
          return [&]<auto... Is>(std::index_sequence<Is...>) {
            return callback(py::cast<std::size_t>(args[Is])...);
          }(std::make_index_sequence<rank>{});
        });
  };

  constexpr auto with_element_and_extents =
      [=](tt::DType dtype, const py::args &args, auto callback) {
        return with_element(dtype, [&](auto element) {
          return with_extents(args, [&](auto... extents) {
            return callback(element, extents...);
          });
        });
      };

  m.def(
      "ones",
      [=](const py::args &args, std::optional<tt::DType> dtype) {
        return with_element_and_extents(
            dtype ? *dtype : *default_dtype, args,
            []<class TElement>(TElement, auto... extents) {
              return py::cast(tt::ones<TElement>(extents...));
            });
      },
      py::kw_only(), py::arg("dtype") = py::none());

  m.def(
      "zeros",
      [=](const py::args &args, std::optional<tt::DType> dtype) {
        return with_element_and_extents(
            dtype ? *dtype : *default_dtype, args,
            []<class TElement>(TElement, auto... extents) {
              return py::cast(tt::zeros<TElement>(extents...));
            });
      },
      py::kw_only(), py::arg("dtype") = py::none());
}
