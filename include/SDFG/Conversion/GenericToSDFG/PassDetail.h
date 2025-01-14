// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// Header for Generic to SDFG conversion pass details.

#ifndef SDFG_Conversion_GenericToSDFG_PassDetail_H
#define SDFG_Conversion_GenericToSDFG_PassDetail_H

#include "SDFG/Dialect/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sdfg {
namespace conversion {

/// Generate the code for base classes.
#define GEN_PASS_CLASSES
#include "SDFG/Conversion/GenericToSDFG/Passes.h.inc"

} // namespace conversion
} // namespace sdfg
} // end namespace mlir

#endif // SDFG_Conversion_GenericToSDFG_PassDetail_H
