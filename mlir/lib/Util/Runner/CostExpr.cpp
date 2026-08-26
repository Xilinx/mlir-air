//===- CostExpr.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_RUNNER_COST_EXPR
#define AIR_UTIL_RUNNER_COST_EXPR

#include "air/Util/Runner.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>

namespace xilinx {
namespace air {

// A cycle count written as an expression over the operands of the op being
// costed, so a machine model can describe a cost the runner's built-in
// throughput formula cannot.
//
// The built-in formula is ops divided by a rate. That is the right shape for a
// vector unit and the wrong shape for anything else. A weight-stationary
// block that streams an activation through fixed weights costs time
// proportional to the weight bit-planes and not at all to the nominal MAC
// count: `4 + 12*bits1`. A block that processes a tile at a time costs
// `ceildiv(volume1, 4096) * 70`. Neither is expressible as a rate, and both
// are ordinary hardware.
//
// Grammar:
//
//   expr    := term (('+' | '-') term)*
//   term    := factor (('*' | '/' | '%') factor)*
//   factor  := '-'? primary
//   primary := number | ident | ident '(' expr (',' expr)* ')' | '(' expr ')'
//
// Functions: ceil, floor, min, max, ceildiv.
//
// Variables are supplied by the caller; see costExprVariables in Runner.cpp for
// the set and what each means.
class costExpr {
public:
  using VarMap = llvm::StringMap<double>;

  costExpr(llvm::StringRef text) : text(text.str()) {}

  // Evaluate, or return the parse/lookup error.
  llvm::Expected<double> evaluate(const VarMap &vars) const {
    parseState st{text, 0, &vars, ""};
    skipSpace(st);
    double v = parseExpr(st);
    if (st.error.empty()) {
      skipSpace(st);
      if (st.pos != st.s.size())
        st.error = "unexpected '" + st.s.substr(st.pos) + "'";
    }
    if (!st.error.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "in cost expression \"" + text +
                                         "\": " + st.error);
    return v;
  }

private:
  std::string text;

  struct parseState {
    std::string s;
    size_t pos;
    const VarMap *vars;
    std::string error;
  };

  static void skipSpace(parseState &st) {
    while (st.pos < st.s.size() && isspace((unsigned char)st.s[st.pos]))
      st.pos++;
  }

  static bool eat(parseState &st, char c) {
    skipSpace(st);
    if (st.pos < st.s.size() && st.s[st.pos] == c) {
      st.pos++;
      return true;
    }
    return false;
  }

  static double parseExpr(parseState &st) {
    double lhs = parseTerm(st);
    while (st.error.empty()) {
      skipSpace(st);
      if (eat(st, '+'))
        lhs += parseTerm(st);
      else if (eat(st, '-'))
        lhs -= parseTerm(st);
      else
        break;
    }
    return lhs;
  }

  static double parseTerm(parseState &st) {
    double lhs = parseFactor(st);
    while (st.error.empty()) {
      skipSpace(st);
      if (eat(st, '*'))
        lhs *= parseFactor(st);
      else if (eat(st, '/')) {
        double rhs = parseFactor(st);
        if (rhs == 0) {
          st.error = "division by zero";
          return 0;
        }
        lhs /= rhs;
      } else if (eat(st, '%')) {
        double rhs = parseFactor(st);
        if (rhs == 0) {
          st.error = "modulo by zero";
          return 0;
        }
        lhs = std::fmod(lhs, rhs);
      } else
        break;
    }
    return lhs;
  }

  static double parseFactor(parseState &st) {
    skipSpace(st);
    if (eat(st, '-'))
      return -parseFactor(st);
    return parsePrimary(st);
  }

  static double parsePrimary(parseState &st) {
    skipSpace(st);
    if (st.pos >= st.s.size()) {
      st.error = "unexpected end of expression";
      return 0;
    }
    if (eat(st, '(')) {
      double v = parseExpr(st);
      if (!eat(st, ')') && st.error.empty())
        st.error = "expected ')'";
      return v;
    }
    char c = st.s[st.pos];
    if (isdigit((unsigned char)c) || c == '.') {
      // Hand the whole numeric literal to strtod, exponent and all, rather
      // than stopping at the 'e' and leaving it to be read as a variable.
      const char *begin = st.s.c_str() + st.pos;
      char *end = nullptr;
      double v = strtod(begin, &end);
      if (end == begin) {
        st.error = "malformed number";
        return 0;
      }
      st.pos += (size_t)(end - begin);
      return v;
    }
    if (isalpha((unsigned char)c) || c == '_') {
      size_t start = st.pos;
      while (st.pos < st.s.size() &&
             (isalnum((unsigned char)st.s[st.pos]) || st.s[st.pos] == '_'))
        st.pos++;
      std::string name = st.s.substr(start, st.pos - start);
      skipSpace(st);
      if (st.pos < st.s.size() && st.s[st.pos] == '(')
        return parseCall(st, name);
      auto it = st.vars->find(name);
      if (it == st.vars->end()) {
        st.error = "unknown variable '" + name + "'";
        return 0;
      }
      return it->second;
    }
    st.error = std::string("unexpected character '") + c + "'";
    return 0;
  }

  static double parseCall(parseState &st, llvm::StringRef name) {
    llvm::SmallVector<double, 4> args;
    if (!eat(st, '(')) {
      st.error = "expected '('";
      return 0;
    }
    if (!eat(st, ')')) {
      do {
        args.push_back(parseExpr(st));
        if (!st.error.empty())
          return 0;
      } while (eat(st, ','));
      if (!eat(st, ')')) {
        st.error = "expected ')'";
        return 0;
      }
    }
    auto arity = [&](unsigned n) {
      if (args.size() != n) {
        st.error = (name + " takes " + llvm::Twine(n) + " argument(s), got " +
                    llvm::Twine(args.size()))
                       .str();
        return false;
      }
      return true;
    };
    if (name == "ceil")
      return arity(1) ? std::ceil(args[0]) : 0;
    if (name == "floor")
      return arity(1) ? std::floor(args[0]) : 0;
    if (name == "min")
      return arity(2) ? std::min(args[0], args[1]) : 0;
    if (name == "max")
      return arity(2) ? std::max(args[0], args[1]) : 0;
    if (name == "ceildiv") {
      if (!arity(2))
        return 0;
      if (args[1] == 0) {
        st.error = "ceildiv by zero";
        return 0;
      }
      return std::ceil(args[0] / args[1]);
    }
    st.error = "unknown function '" + name.str() +
               "' (have ceil, floor, min, max, ceildiv)";
    return 0;
  }
}; // costExpr

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_COST_EXPR
