#pragma once

#include <tt/core/concepts.hpp>

#include <format>

template <tt::tensor TInput>
struct std::formatter<TInput> {
private:
  using element_type = decltype(+std::declval<tt::element_type_t<TInput>>());

  std::formatter<element_type> element_formatter{};

public:
  constexpr std::format_parse_context::iterator
  parse(std::format_parse_context &ctx) {
    return element_formatter.parse(ctx);
  }

  constexpr std::format_context::iterator
  format(const TInput input, std::format_context &ctx) const {
    auto out = std::format_to(ctx.out(), "[");

    const auto recur = [&, &element_formatter = element_formatter](
                           const auto &recur, const auto... indices) {
      constexpr auto rank = sizeof...(indices);

      for (std::size_t index = 0; index < input.extent(rank); ++index) {
        if constexpr (rank + 1 == TInput::rank()) {
          if (index > 0) {
            out = std::format_to(out, ",");
          }

          out = element_formatter.format(+input(indices..., index), ctx);
        } else {
          if (index > 0) {
            // indentation
            out = std::format_to(out, ",\n{:<{}}", "", rank + 1);
          }

          out = std::format_to(out, "[");
          recur(recur, indices..., index);
          out = std::format_to(out, "]");
        }
      }
    };

    recur(recur);

    return std::format_to(out, "]");
  }
};
