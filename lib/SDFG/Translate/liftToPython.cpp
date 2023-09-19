// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains a Python lifter, which lifts MLIR operations to Python
/// code.

#include "SDFG/Translate/liftToPython.h"
#include "SDFG/Utils/Utils.h"

using namespace mlir;
using namespace sdfg;

std::string getMaxValueForBitwidth(unsigned bitwidth) {
  return "2**" + std::to_string(bitwidth) + " - 1";
}

std::string getMaxValueForSignedBitwidth(unsigned bitwidth) {
  return "2**(" + std::to_string(bitwidth - 1) + ") - 1";
}

std::string getMinValueForSignedBitwidth(unsigned bitwidth) {
  return "-2**(" + std::to_string(bitwidth - 1) + ")";
}

unsigned getBitwidth(Type t) {
  if (t.isIntOrFloat())
    return t.getIntOrFloatBitWidth();
  // e.g. index
  return 64;
}

// TODO(later): Temporary auto-lifting. Will be included into DaCe
/// Converts a single operation to a single line of Python code. If successful,
/// returns Python code as s string.
Optional<std::string> liftOperationToPython(Operation &op, Operation &source) {
  //===--------------------------------------------------------------------===//
  // Arith
  //===--------------------------------------------------------------------===//

  if (isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " + " + nameArg1;
  }

  if (isa<arith::SubFOp>(op) || isa<arith::SubIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " - " + nameArg1;
  }

  if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " * " + nameArg1;
  }

  if (isa<arith::DivFOp>(op) || isa<arith::DivSIOp>(op) ||
      isa<arith::DivUIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = " + nameArg0 + " / " + nameArg1;
  }

  if (arith::NegFOp negFOp = dyn_cast<arith::NegFOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = -" +
           sdfg::utils::valueToString(negFOp.getOperand(), op);
  }

  if (isa<arith::RemSIOp>(op) || isa<arith::RemUIOp>(op) ||
      isa<arith::RemFOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = " + sdfg::utils::valueToString(op.getOperand(0), op) +
           " % " + sdfg::utils::valueToString(op.getOperand(1), op);
  }

  if (arith::IndexCastOp indexCast = dyn_cast<arith::IndexCastOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = " + sdfg::utils::valueToString(indexCast.getIn(), op);
  }

  if (isa<arith::SIToFPOp>(op) || isa<arith::UIToFPOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = float(" +
           sdfg::utils::valueToString(op.getOperand(0), op) + ")";
  }

  if (isa<arith::FPToSIOp>(op) || isa<arith::FPToUIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = int(" +
           sdfg::utils::valueToString(op.getOperand(0), op) + ")";
  }

  if (isa<arith::MaxFOp>(op) || isa<arith::MaxSIOp>(op) ||
      isa<arith::MaxUIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = max(" +
           sdfg::utils::valueToString(op.getOperand(0), op) + ", " +
           sdfg::utils::valueToString(op.getOperand(1), op) + ")";
  }

  if (isa<arith::CmpIOp>(op) || isa<arith::CmpFOp>(op)) {
    Value lhsValue;
    Value rhsValue;
    std::string predicate = "";

    if (isa<arith::CmpIOp>(op)) {
      arith::CmpIOp cmp = dyn_cast<arith::CmpIOp>(op);
      lhsValue = cmp.getLhs();
      rhsValue = cmp.getRhs();

      switch (cmp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        predicate = "==";
        break;

      case arith::CmpIPredicate::ne:
        predicate = "!=";
        break;

      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        predicate = ">=";
        break;

      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        predicate = ">";
        break;

      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        predicate = "<=";
        break;

      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        predicate = "<";
        break;

      default:
        break;
      }
    }

    else {
      arith::CmpFOp cmp = dyn_cast<arith::CmpFOp>(op);
      lhsValue = cmp.getLhs();
      rhsValue = cmp.getRhs();

      switch (cmp.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ:
        predicate = "==";
        break;

      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE:
        predicate = "!=";
        break;

      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE:
        predicate = ">=";
        break;

      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT:
        predicate = ">";
        break;

      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE:
        predicate = "<=";
        break;

      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT:
        predicate = "<";
        break;

      default:
        break;
      }
    }

    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(lhsValue, op);
    std::string rhs = sdfg::utils::valueToString(rhsValue, op);
    return nameOut + " = " + lhs + " " + predicate + " " + rhs;
  }

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

    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = " + val;
  }

  if (arith::SelectOp selectOp = dyn_cast<arith::SelectOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = " +
           sdfg::utils::valueToString(selectOp.getTrueValue(), op) + " if " +
           sdfg::utils::valueToString(selectOp.getCondition(), op) + " else " +
           sdfg::utils::valueToString(selectOp.getFalseValue(), op);
  }

  if (isa<arith::ExtSIOp>(op) || isa<arith::ExtUIOp>(op) ||
      isa<arith::ExtFOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = " + sdfg::utils::valueToString(op.getOperand(0), op);
  }

  if (arith::OrIOp oriOp = dyn_cast<arith::OrIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(oriOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(oriOp.getOperand(1), op);
    return nameOut + " = " + lhs + " | " + rhs;
  }

  if (arith::AndIOp andiOp = dyn_cast<arith::AndIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(andiOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(andiOp.getOperand(1), op);
    return nameOut + " = " + lhs + " & " + rhs;
  }

  if (arith::BitcastOp bitcastOp = dyn_cast<arith::BitcastOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = " +
           sdfg::utils::valueToString(bitcastOp.getOperand(), op);
  }

  if (arith::ShLIOp shliOp = dyn_cast<arith::ShLIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(shliOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(shliOp.getOperand(1), op);
    return nameOut + " = " + lhs + " << " + rhs;
  }

  if (arith::ShRSIOp shrsiOp = dyn_cast<arith::ShRSIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(shrsiOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(shrsiOp.getOperand(1), op);
    return nameOut + " = " + lhs + " >> " + rhs;
  }

  if (arith::ShRUIOp shruiOp = dyn_cast<arith::ShRUIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(shruiOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(shruiOp.getOperand(1), op);
    return nameOut + " = " + lhs + " >> " + rhs;
  }

  if (arith::CeilDivUIOp ceildivuiOp = dyn_cast<arith::CeilDivUIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string numerator =
        sdfg::utils::valueToString(ceildivuiOp.getOperand(0), op);
    std::string denominator =
        sdfg::utils::valueToString(ceildivuiOp.getOperand(1), op);
    return nameOut + " = -(-" + numerator + " // " + denominator + ")";
  }

  if (arith::XOrIOp xoriOp = dyn_cast<arith::XOrIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(xoriOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(xoriOp.getOperand(1), op);
    return nameOut + " = " + lhs + " ^ " + rhs;
  }

  if (arith::MinUIOp minuiOp = dyn_cast<arith::MinUIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(minuiOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(minuiOp.getOperand(1), op);
    return nameOut + " = min(" + lhs + ", " + rhs + ")";
  }

  if (arith::MinSIOp minuiOp = dyn_cast<arith::MinSIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(minuiOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(minuiOp.getOperand(1), op);
    return nameOut + " = min(" + lhs + ", " + rhs + ")";
  }

  if (arith::IndexCastUIOp index_castuiOp =
          dyn_cast<arith::IndexCastUIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand =
        sdfg::utils::valueToString(index_castuiOp.getOperand(), op);
    return nameOut + " = int(" + operand + ")";
  }

  if (arith::MinFOp minfOp = dyn_cast<arith::MinFOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs = sdfg::utils::valueToString(minfOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(minfOp.getOperand(1), op);
    return nameOut + " = min(" + lhs + ", " + rhs + ")";
  }

  if (arith::FloorDivSIOp floordivsiOp = dyn_cast<arith::FloorDivSIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string lhs =
        sdfg::utils::valueToString(floordivsiOp.getOperand(0), op);
    std::string rhs =
        sdfg::utils::valueToString(floordivsiOp.getOperand(1), op);
    return nameOut + " = " + lhs + " // " + rhs;
  }

  if (arith::CeilDivSIOp ceildivsiOp = dyn_cast<arith::CeilDivSIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string numerator =
        sdfg::utils::valueToString(ceildivsiOp.getOperand(0), op);
    std::string denominator =
        sdfg::utils::valueToString(ceildivsiOp.getOperand(1), op);
    return nameOut + " = math.ceil(" + numerator + " / " + denominator + ")";
  }

  if (arith::AddUIExtendedOp addOp = dyn_cast<arith::AddUIExtendedOp>(op)) {
    unsigned bitwidth = getBitwidth(addOp.getType(0));
    std::string nameOutSum = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameOutOverflow =
        sdfg::utils::valueToString(op.getResult(1), op);
    std::string lhs = sdfg::utils::valueToString(addOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(addOp.getOperand(1), op);

    std::string sumExpression = lhs + " + " + rhs;
    std::string overflowExpression =
        "(" + sumExpression + " > " + getMaxValueForBitwidth(bitwidth) + ")";
    return nameOutSum + " = " + sumExpression + "\\n" + nameOutOverflow +
           " = " + overflowExpression;
  }

  if (arith::MulSIExtendedOp mulOp = dyn_cast<arith::MulSIExtendedOp>(op)) {
    unsigned bitwidth = getBitwidth(mulOp.getType(0));
    std::string nameOutLow = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameOutHigh = sdfg::utils::valueToString(op.getResult(1), op);
    std::string lhs = sdfg::utils::valueToString(mulOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(mulOp.getOperand(1), op);

    std::string divValue = "(" + getMaxValueForSignedBitwidth(bitwidth) +
                           " + 1 if " + lhs + " * " + rhs + " >= 0 else " +
                           getMinValueForSignedBitwidth(bitwidth) + ")";
    return nameOutLow + ", " + nameOutHigh + " = divmod(" + lhs + " * " + rhs +
           ", " + divValue + ")";
  }
  if (arith::MulUIExtendedOp mulOp = dyn_cast<arith::MulUIExtendedOp>(op)) {
    unsigned bitwidth = getBitwidth(mulOp.getType(0));

    std::string nameOutLow = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameOutHigh = sdfg::utils::valueToString(op.getResult(1), op);
    std::string lhs = sdfg::utils::valueToString(mulOp.getOperand(0), op);
    std::string rhs = sdfg::utils::valueToString(mulOp.getOperand(1), op);

    return nameOutLow + ", " + nameOutHigh + " = divmod(" + lhs + " * " + rhs +
           ", " + getMaxValueForBitwidth(bitwidth) + " + 1)";
  }

  if (arith::CeilDivSIOp divOp = dyn_cast<arith::CeilDivSIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string numerator = sdfg::utils::valueToString(divOp.getOperand(0), op);
    std::string denominator =
        sdfg::utils::valueToString(divOp.getOperand(1), op);
    return nameOut + " = math.ceil(" + numerator + " / " + denominator + ")";
  }

  if (arith::TruncIOp truncOp = dyn_cast<arith::TruncIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(truncOp.getOperand(), op);
    return nameOut + " = int(" + operand + ")";
  }

  if (arith::TruncFOp truncOp = dyn_cast<arith::TruncFOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(truncOp.getOperand(), op);
    return nameOut + " = math.trunc(" + operand + ")";
  }

  if (arith::IndexCastOp castOp = dyn_cast<arith::IndexCastOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(castOp.getOperand(), op);
    return nameOut + " = int(" + operand + ")";
  }

  //===--------------------------------------------------------------------===//
  // Math
  //===--------------------------------------------------------------------===//

  if (math::SqrtOp sqrtOp = dyn_cast<math::SqrtOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = math.sqrt(" +
           sdfg::utils::valueToString(sqrtOp.getOperand(), op) + ")";
  }

  if (math::ExpOp expOp = dyn_cast<math::ExpOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = math.exp(" +
           sdfg::utils::valueToString(expOp.getOperand(), op) + ")";
  }

  if (math::PowFOp powFOp = dyn_cast<math::PowFOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameArg0 = sdfg::utils::valueToString(op.getOperand(0), op);
    std::string nameArg1 = sdfg::utils::valueToString(op.getOperand(1), op);
    return nameOut + " = math.pow(" + nameArg0 + "," + nameArg1 + ")";
  }

  if (math::CosOp cosOp = dyn_cast<math::CosOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = math.cos(" +
           sdfg::utils::valueToString(cosOp.getOperand(), op) + ")";
  }

  if (math::SinOp sinOp = dyn_cast<math::SinOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = math.sin(" +
           sdfg::utils::valueToString(sinOp.getOperand(), op) + ")";
  }

  if (math::LogOp logOp = dyn_cast<math::LogOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = math.log(" +
           sdfg::utils::valueToString(logOp.getOperand(), op) + ")";
  }

  if (math::CountTrailingZerosOp cttzOp =
          dyn_cast<math::CountTrailingZerosOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(cttzOp.getOperand(), op);
    return nameOut + " = (" + operand + " & -" + operand + ").bit_count()";
  }

  if (math::CountLeadingZerosOp ctlzOp =
          dyn_cast<math::CountLeadingZerosOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(ctlzOp.getOperand(), op);
    return nameOut + " = (len(bin(" + operand + ")) - len(bin(" + operand +
           ").lstrip('0')) - 1)";
  }

  if (math::Log2Op log2Op = dyn_cast<math::Log2Op>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(log2Op.getOperand(), op);
    return nameOut + " = math.log2(" + operand + ")";
  }

  if (math::RsqrtOp rsqrtOp = dyn_cast<math::RsqrtOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(rsqrtOp.getOperand(), op);
    return nameOut + " = 1 / math.sqrt(" + operand + ")";
  }

  if (math::ErfOp erfOp = dyn_cast<math::ErfOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(erfOp.getOperand(), op);
    return nameOut + " = math.erf(" + operand + ")";
  }

  if (math::Exp2Op exp2Op = dyn_cast<math::Exp2Op>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(exp2Op.getOperand(), op);
    return nameOut + " = math.exp2(" + operand + ")";
  }

  if (math::IPowIOp ipowiOp = dyn_cast<math::IPowIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string base = sdfg::utils::valueToString(ipowiOp.getOperand(0), op);
    std::string exponent =
        sdfg::utils::valueToString(ipowiOp.getOperand(1), op);
    return nameOut + " = " + base + " ** " + exponent;
  }

  if (math::TruncOp truncOp = dyn_cast<math::TruncOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(truncOp.getOperand(), op);
    return nameOut + " = math.trunc(" + operand + ")";
  }

  if (math::Log10Op log10Op = dyn_cast<math::Log10Op>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(log10Op.getOperand(), op);
    return nameOut + " = math.log10(" + operand + ")";
  }

  if (math::Log1pOp log1pOp = dyn_cast<math::Log1pOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(log1pOp.getOperand(), op);
    return nameOut + " = math.log1p(" + operand + ")";
  }

  if (math::AbsIOp absiOp = dyn_cast<math::AbsIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(absiOp.getOperand(), op);
    return nameOut + " = abs(" + operand + ")";
  }

  if (math::CbrtOp cbrtOp = dyn_cast<math::CbrtOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(cbrtOp.getOperand(), op);
    return nameOut + " = math.pow(" + operand + ", 1/3)";
  }

  if (math::TanOp tanOp = dyn_cast<math::TanOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(tanOp.getOperand(), op);
    return nameOut + " = math.tan(" + operand + ")";
  }

  if (math::CtPopOp ctpopOp = dyn_cast<math::CtPopOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(ctpopOp.getOperand(), op);
    return nameOut + " = bin(" + operand + ").count('1')";
  }

  if (math::FmaOp fmaOp = dyn_cast<math::FmaOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string x = sdfg::utils::valueToString(fmaOp.getOperand(0), op);
    std::string y = sdfg::utils::valueToString(fmaOp.getOperand(1), op);
    std::string z = sdfg::utils::valueToString(fmaOp.getOperand(2), op);
    return nameOut + " = math.fma(" + x + ", " + y + ", " + z + ")";
  }

  if (math::FloorOp floorOp = dyn_cast<math::FloorOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(floorOp.getOperand(), op);
    return nameOut + " = math.floor(" + operand + ")";
  }

  if (math::ExpM1Op expm1Op = dyn_cast<math::ExpM1Op>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(expm1Op.getOperand(), op);
    return nameOut + " = math.expm1(" + operand + ")";
  }

  if (math::AbsFOp absOp = dyn_cast<math::AbsFOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(absOp.getOperand(), op);
    return nameOut + " = abs(" + operand + ")";
  }

  if (math::AtanOp atanOp = dyn_cast<math::AtanOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(atanOp.getOperand(), op);
    return nameOut + " = math.atan(" + operand + ")";
  }

  if (math::Atan2Op atan2Op = dyn_cast<math::Atan2Op>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string y = sdfg::utils::valueToString(atan2Op.getOperand(0), op);
    std::string x = sdfg::utils::valueToString(atan2Op.getOperand(1), op);
    return nameOut + " = math.atan2(" + y + ", " + x + ")";
  }

  if (math::CeilOp ceilOp = dyn_cast<math::CeilOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(ceilOp.getOperand(), op);
    return nameOut + " = math.ceil(" + operand + ")";
  }

  if (math::CopySignOp copySignOp = dyn_cast<math::CopySignOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string x = sdfg::utils::valueToString(copySignOp.getOperand(0), op);
    std::string y = sdfg::utils::valueToString(copySignOp.getOperand(1), op);
    return nameOut + " = math.copysign(" + x + ", " + y + ")";
  }

  if (math::TanhOp tanhOp = dyn_cast<math::TanhOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(tanhOp.getOperand(), op);
    return nameOut + " = math.tanh(" + operand + ")";
  }

  if (math::RoundEvenOp roundEvenOp = dyn_cast<math::RoundEvenOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand =
        sdfg::utils::valueToString(roundEvenOp.getOperand(), op);
    return nameOut + " = round(" + operand + ")";
  }

  if (math::RoundOp roundOp = dyn_cast<math::RoundOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string operand = sdfg::utils::valueToString(roundOp.getOperand(), op);
    return nameOut + " = round(" + operand + ")";
  }

  if (math::FPowIOp powOp = dyn_cast<math::FPowIOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string base = sdfg::utils::valueToString(powOp.getOperand(0), op);
    std::string exponent = sdfg::utils::valueToString(powOp.getOperand(1), op);
    return nameOut + " = " + base + " ** " + exponent;
  }

  //===--------------------------------------------------------------------===//
  // LLVM
  //===--------------------------------------------------------------------===//

  if (isa<mlir::LLVM::UndefOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = -1";
  }

  //===--------------------------------------------------------------------===//
  // SDFG
  //===--------------------------------------------------------------------===//

  if (SymOp sym = dyn_cast<SymOp>(op)) {
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = " + sym.getExpr().str();
  }

  if (StoreOp store = dyn_cast<StoreOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(sdfg::utils::valueToString(op.getOperand(i), op));
    }

    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameVal = sdfg::utils::valueToString(store.getArr(), op);
    return nameOut + "[" + indices + "]" + " = " + nameVal;
  }

  if (LoadOp load = dyn_cast<LoadOp>(op)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumOperands() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(sdfg::utils::valueToString(op.getOperand(i), op));
    }

    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    std::string nameArr = sdfg::utils::valueToString(load.getArr(), op);
    return nameOut + " = " + nameArr + "[" + indices + "]";
  }

  if (StreamLengthOp streamLen = dyn_cast<StreamLengthOp>(op)) {
    // FIXME: What's the proper stream name?
    std::string streamName = sdfg::utils::valueToString(streamLen.getStr(), op);
    std::string nameOut = sdfg::utils::valueToString(op.getResult(0), op);
    return nameOut + " = len(" + streamName + ")";
  }

  if (isa<sdfg::ReturnOp>(op)) {
    return "";
  }

  //===--------------------------------------------------------------------===//
  // Func
  //===--------------------------------------------------------------------===//

  if (isa<func::ReturnOp>(op)) {
    std::string code = "";
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (i > 0)
        code.append("\\n");
      // FIXME: What's the proper return name?
      code.append("_out = " + sdfg::utils::valueToString(op.getOperand(i), op));
    }
    return code;
  }

  return std::nullopt;
}

/// Converts the operations in the first region of op to Python code. If
/// successful, returns Python code as a string.
Optional<std::string> translation::liftToPython(Operation &op) {
  std::string code = "";

  for (Operation &oper : op.getRegion(0).getOps()) {
    Optional<std::string> line = liftOperationToPython(oper, op);
    if (line.has_value()) {
      code.append(line.value() + "\\n");
    } else {
      emitRemark(op.getLoc(), "No lifting to python possible");
      emitRemark(oper.getLoc(), "Failed to lift");
      return std::nullopt;
    }
  }

  return code;
}

/// Provides a name for the tasklet.
std::string translation::getTaskletName(Operation &op) {
  Operation &firstOp = *op.getRegion(0).getOps().begin();
  return sdfg::utils::operationToString(firstOp);
}
