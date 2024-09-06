#pragma once

#if __cplusplus >= 202002L

#define TT_CONCEPT concept
#define TT_CONSTEVAL consteval
#define TT_EXPLICIT(VALUE) explicit(VALUE)

#else

#define TT_CONCEPT inline constexpr bool
#define TT_CONSTEVAL constexpr

#define _TT_EXPLICIT_false()
#define _TT_EXPLICIT_true() explicit
#define _TT_CAT(A, B) A##B
#define TT_CAT(A, B) _TT_CAT(A, B)
#define TT_EXPLICIT(VALUE) TT_CAT(_TT_EXPLICIT_, VALUE)()

#endif

#define TT_REQUIRES(...) ::std::enable_if_t<__VA_ARGS__>
