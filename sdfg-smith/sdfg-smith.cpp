// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the SDFG program generator.

#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-smith/MlirSmithMain.h"

#include "SDFG/Dialect/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::sdfg::SDFGDialect>();

  return mlir::failed(mlir::mlirSmithMain(argc, argv, registry,
                                          mlir::sdfg::SDFGNode::generate));
}
