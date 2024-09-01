#include <tt/core/bfloat16.hpp>
#include <tt/core/dtype.hpp>
#include <tt/core/format.hpp>
#include <tt/core/shared_tensor.hpp>
#include <tt/core/tiled_layout.hpp>
#include <tt/operators/arange.hpp>
#include <tt/operators/full.hpp>
#include <tt/operators/ones.hpp>
#include <tt/operators/zeros.hpp>

#include <boost/mp11.hpp>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <format>
#include <iostream>

constexpr auto name_of(tt::row_major_layout) { return "row_major"; }
constexpr auto name_of(tt::strided_layout) { return "strided"; }
constexpr auto name_of(tt::tiled_layout) { return "tiled"; }

constexpr auto name_of(tt::dtype_t<tt::float32>) { return "float32"; }
constexpr auto name_of(tt::dtype_t<tt::float64>) { return "float64"; }
constexpr auto name_of(tt::dtype_t<tt::bfloat16>) { return "bfloat16"; }
constexpr auto name_of(tt::dtype_t<std::uint8_t>) { return "uint8"; }
constexpr auto name_of(tt::dtype_t<std::int8_t>) { return "int8"; }
constexpr auto name_of(tt::dtype_t<std::int16_t>) { return "int16"; }
constexpr auto name_of(tt::dtype_t<std::int32_t>) { return "int32"; }
constexpr auto name_of(tt::dtype_t<std::int64_t>) { return "int64"; }
constexpr auto name_of(tt::dtype_t<bool>) { return "bool"; }

template <class TElement, class TExtents, class TLayout>
inline constexpr bool is_tensor_traits_valid_v =
    TExtents::rank() >= 2 or not std::is_same_v<TLayout, tt::tiled_layout>;

template <std::size_t, class T>
using dependent = T;

PYBIND11_MODULE(tt, m) {
  namespace py = pybind11;
  namespace mp = boost::mp11;

  using element_types =
      mp::mp_list<tt::float32, tt::float64, tt::bfloat16, std::uint8_t,
                  std::int8_t, std::int16_t, std::int32_t, std::int64_t, bool>;
  using extents_types =
      mp::mp_list<tt::dims<1>, tt::dims<2>, tt::dims<3>, tt::dims<4>,
                  tt::dims<5>, tt::dims<6>, tt::dims<7>, tt::dims<8>>;
  using layout_types =
      mp::mp_list<tt::row_major_layout, tt::strided_layout, tt::tiled_layout>;
  using tensor_traits =
      mp::mp_product<mp::mp_list, element_types, extents_types, layout_types>;

  const auto m_dtype = m.def_submodule("dtype");
  const auto m_tensors = m.def_submodule("tensors");

  mp::mp_for_each<element_types>([&]<class TElement>(TElement) {
    py::class_<tt::dtype_t<TElement>> type(m_dtype,
                                           name_of(tt::dtype<TElement>));
  });

  const auto register_tensor_for =
      [&]<class TElement, class TExtents, class TLayout>(
          mp::mp_list<TElement, TExtents, TLayout>) {
        using tensor_type = tt::shared_tensor<TElement, TExtents, TLayout>;

        constexpr auto element_name = name_of(tt::dtype<TElement>);
        constexpr auto rank = TExtents::rank();
        constexpr auto layout_name = name_of(TLayout{});

        py::class_<tensor_type> tensor_class{
            m_tensors,
            std::format("{}_{}d_{}_tensor", element_name, rank, layout_name)
                .c_str()};

        [&]<auto... Is>(std::index_sequence<Is...>) {
          const std::array<py::arg, rank> args{
              py::arg(std::format("d{}", Is).c_str())...};

          tensor_class.def(
              "__getitem__", [=](const tensor_type &self, py::tuple key) {
                if (key.size() != rank) {
                  throw py::index_error("Invalid index");
                }

                return self(key[Is].template cast<std::size_t>()...);
              });

          tensor_class.def("__setitem__", [=](tensor_type &self, py::tuple key,
                                              TElement value) {
            if (key.size() != rank) {
              throw py::index_error("Invalid index");
            }

            return self(key[Is].template cast<std::size_t>()...) = value;
          });

          if constexpr (std::is_same_v<TLayout, tt::row_major_layout>) {
            using dtype_class = py::class_<tt::dtype_t<TElement>>;

            const py::arg dtype(
                std::format("dtype_{}", name_of(tt::dtype<TElement>)).c_str());

            m.def(
                "empty",
                [](dtype_class, dependent<Is, std::size_t>... extents) {
                  return tt::empty<TElement>(extents...);
                },
                dtype, args[Is]...);

            m.def(
                "full",
                [](TElement fill_value, dependent<Is, std::size_t>... extents) {
                  return tt::full(fill_value, extents...);
                },
                py::arg("fill_value"), args[Is]...);

            m.def(
                "ones",
                [](dtype_class, dependent<Is, std::size_t>... extents) {
                  return tt::ones<TElement>(extents...);
                },
                dtype, args[Is]...);

            m.def(
                "zeros",
                [](dtype_class, dependent<Is, std::size_t>... extents) {
                  return tt::zeros<TElement>(extents...);
                },
                dtype, args[Is]...);
          }
        }(std::make_index_sequence<rank>{});

        m.def(
            "print",
            [](std::string_view fmt, const tensor_type &tensor_arg) {
              std::vformat_to(std::ostreambuf_iterator<char>(std::cout), fmt,
                              std::make_format_args(tensor_arg));
            },
            py::arg("fmt"), py::arg("tensor_arg"));
      };

  mp::mp_for_each<tensor_traits>(
      [=]<class TElement, class TExtents, class TLayout>(
          mp::mp_list<TElement, TExtents, TLayout> traits) {
        if constexpr (is_tensor_traits_valid_v<TElement, TExtents, TLayout>) {
          register_tensor_for(traits);
        }
      });

  mp::mp_for_each<element_types>([&]<class TElement>(TElement) {
    m.def(
        "arange",
        [](TElement start, TElement end, TElement step) {
          return tt::arange<TElement>(start, end, step);
        },
        py::arg("start") = 0, py::arg("end"), py::arg("step") = 1);
  });
}
