#ifndef SDFG_Utils_ValueToString_H
#define SDFG_Utils_ValueToString_H

#include "mlir/IR/Value.h"
#include <string>

namespace mlir::sdfg::utils {

std::string valueToString(Value value, bool useSDFG = false);
std::string valueToString(Value value, Operation &op, bool useSDFG = false);

} // namespace mlir::sdfg::utils

#endif // SDFG_Utils_ValueToString_H
