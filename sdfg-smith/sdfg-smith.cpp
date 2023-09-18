// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the SDFG program generator.

#include "SDFG/Dialect/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-smith/MlirSmithMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::sdfg::SDFGDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::math::MathDialect>();

  return mlir::failed(mlir::mlirSmithMain(argc, argv, registry,
                                          mlir::sdfg::SDFGNode::generate));
}
