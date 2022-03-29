#include "SDFG/Translate/liftToPython.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace sdfg;

// TODO(later): Temporary auto-lifting. Will be included into DaCe
Optional<std::string> liftOperationToPython(Operation &op, TaskletNode &task) {
  std::string nameOut =
      op.getNumResults() == 1 ? utils::valueToString(op.getResult(0), op) : "";

  if (isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op)) {
    std::string nameArg0 = utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " + " + nameArg1;
  }

  if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
    std::string nameArg0 = utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " * " + nameArg1;
  }

  if (isa<arith::IndexCastOp>(op)) {
    return nameOut + " = " + utils::valueToString(op.getOperand(0), op);
  }

  if (SymOp sym = dyn_cast<SymOp>(op)) {
    return nameOut + " = " + sym.expr().str();
  }

  // TODO: Add arith ops

  if (isa<arith::ConstantOp>(op)) {
    std::string val;

    if (arith::ConstantFloatOp flop = dyn_cast<arith::ConstantFloatOp>(op)) {
      SmallVector<char> flopVec;
      flop.value().toString(flopVec);
      for (char c : flopVec)
        val += c;
    } else if (arith::ConstantIntOp iop = dyn_cast<arith::ConstantIntOp>(op)) {
      val = std::to_string(iop.value());
    } else if (arith::ConstantIndexOp iop =
                   dyn_cast<arith::ConstantIndexOp>(op)) {
      val = std::to_string(iop.value());
    }

    return nameOut + " = " + val;
  }

  if (isa<StoreOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(utils::valueToString(op.getOperand(i), op));
    }

    std::string nameVal =
        utils::valueToString(op.getOperand(op.getNumOperands() - 1), op);
    return nameOut + "[" + indices + "]" + " = " + nameVal;
  }

  if (isa<LoadOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(utils::valueToString(op.getOperand(i), op));
    }

    std::string nameArr =
        utils::valueToString(op.getOperand(op.getNumOperands() - 1), op);
    return nameOut + " = " + nameArr + "[" + indices + "]";
  }

  // TODO: Handle multiple returns
  if (isa<sdfg::ReturnOp>(op)) {
    std::string code = "";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      code.append(task.getOutputName(i) + " = " +
                  utils::valueToString(op.getOperand(0), op));
    }
    return code;
  }

  return None;
}

// If successful returns Python code as string
Optional<std::string> translation::liftToPython(TaskletNode &op) {
  std::string code = "";

  for (Operation &oper : op.body().getOps()) {
    Optional<std::string> line = liftOperationToPython(oper, op);
    if (line.hasValue()) {
      code.append(line.getValue() + "\\n");
    } else {
      emitRemark(op.getLoc(), "No lifting to python possible");
      return None;
    }
  }

  return code;
}

std::string translation::getTaskletName(TaskletNode &op) {
  Operation &firstOp = *op.body().getOps().begin();

  if (isa<arith::AddFOp>(firstOp) || isa<arith::AddIOp>(firstOp))
    return "add";
  else if (isa<arith::MulFOp>(firstOp) || isa<arith::MulIOp>(firstOp))
    return "mult";
  else if (isa<arith::ConstantOp>(firstOp))
    return "constant";
  else if (isa<arith::IndexCastOp>(firstOp))
    return "cast";
  else if (isa<StoreOp>(firstOp))
    return "store";
  else if (isa<LoadOp>(firstOp))
    return "load";
  else if (isa<SymOp>(firstOp))
    return "sym";
  else if (isa<sdfg::ReturnOp>(firstOp))
    return "return";

  return "task";
}
