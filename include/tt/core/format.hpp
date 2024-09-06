#pragma once

#include <tt/core/concepts.hpp>

#include <fmt/base.h>

#include <string_view>

template <class TInput>
struct fmt::formatter<TInput, char, TT_REQUIRES(tt::tensor<TInput>)> {
private:
  using element_type = decltype(+std::declval<tt::element_type_t<TInput>>());

  fmt::formatter<element_type> element_formatter{};

public:
  constexpr fmt::format_parse_context::iterator
  parse(fmt::format_parse_context &ctx) {
    return element_formatter.parse(ctx);
  }

  constexpr fmt::format_context::iterator
  format(const TInput &input, fmt::format_context &ctx) const {
    if constexpr (TInput::rank() == 0) {
      auto out = fmt::format_to(ctx.out(), "tensor(");
      out = element_formatter.format(+input(), ctx);
      return fmt::format_to(out, ")");
    } else {
      constexpr std::string_view prefix = "tensor([";
      auto out = fmt::format_to(ctx.out(), prefix);

      const auto recur = [&, this](const auto &recur, const auto... indices) {
        constexpr auto rank = sizeof...(indices);

        for (std::size_t index = 0; index < input.extent(rank); ++index) {
          if constexpr (rank + 1 == TInput::rank()) {
            if (index > 0) {
              out = fmt::format_to(out, ", ");
            }

            out = element_formatter.format(+input(indices..., index), ctx);
          } else {
            if (index > 0) {
              constexpr auto indentation = prefix.size() + rank;
              out = fmt::format_to(out, ",\n{:<{}}", "", indentation);
            }

            out = fmt::format_to(out, "[");
            recur(recur, indices..., index);
            out = fmt::format_to(out, "]");
          }
        }
      };

      recur(recur);

      return fmt::format_to(out, "])");
    }
  }
};
