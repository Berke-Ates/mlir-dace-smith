// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the SDFG dialect initializer and the type definitions,
/// such as parsing, printing and utility functions.

#include "SDFG/Dialect/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/GeneratorOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sdfg;

#include "SDFG/Dialect/OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SDFG Dialect
//===----------------------------------------------------------------------===//

/// Initializes the SDFG dialect by adding all operation and type declarations.
void SDFGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SDFG/Dialect/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "SDFG/Dialect/OpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SDFG Types
//===----------------------------------------------------------------------===//

// FIXME: Rewrite to only use an ArrayAttr containing strings & ints
/// Parses a list of dimensions consisting of symbols, constants and question
/// marks.
static ParseResult parseDimensionList(AsmParser &parser, Type &elemType,
                                      SmallVector<StringAttr> &symbols,
                                      SmallVector<int64_t> &integers,
                                      SmallVector<bool> &shape) {
  if (parser.parseLess())
    return failure();

  do {
    OptionalParseResult typeOPR = parser.parseOptionalType(elemType);
    if (typeOPR.has_value() && typeOPR.value().succeeded() &&
        parser.parseGreater().succeeded()) {
      return success();
    }

    if (parser.parseOptionalKeyword("sym").succeeded()) {
      std::string symEx;

      if (parser.parseLParen() || parser.parseString(&symEx) ||
          parser.parseRParen())
        return failure();

      symbols.push_back(parser.getBuilder().getStringAttr(symEx));
      shape.push_back(false);
      continue;
    }

    int64_t num = -1;
    OptionalParseResult intOPR = parser.parseOptionalInteger(num);
    if (intOPR.has_value() && intOPR.value().succeeded()) {
      integers.push_back(num);
      shape.push_back(true);
      continue;
    }

    if (parser.parseOptionalQuestion().succeeded()) {
      integers.push_back(-1);
      shape.push_back(true);
      continue;
    }

    return failure();
  } while (parser.parseXInDimensionList().succeeded());

  return failure();
}

/// Prints a list of dimensions in human-readable form.
static void printDimensionList(AsmPrinter &printer, Type &elemType,
                               ArrayRef<StringAttr> &symbols,
                               ArrayRef<int64_t> &integers,
                               ArrayRef<bool> &shape) {
  unsigned symIdx = 0;
  unsigned intIdx = 0;

  printer << "<";

  for (unsigned i = 0; i < shape.size(); ++i)
    if (shape[i])
      if (integers[intIdx++] == -1)
        printer << "?x";
      else
        printer << integers[intIdx - 1] << "x";
    else
      printer << "sym(" << symbols[symIdx++] << ")x";

  printer << elemType << ">";
}

/// Attempts to parse an array type.
::mlir::Type ArrayType::parse(::mlir::AsmParser &odsParser) {
  Type elementType;
  SmallVector<StringAttr> symbols;
  SmallVector<int64_t> integers;
  SmallVector<bool> shape;
  if (parseDimensionList(odsParser, elementType, symbols, integers, shape))
    return Type();

  SizedType sized = SizedType::get(odsParser.getContext(), elementType, symbols,
                                   integers, shape);
  return get(odsParser.getContext(), sized);
}

/// Prints an array type in human-readable form.
void ArrayType::print(::mlir::AsmPrinter &odsPrinter) const {
  Type elemType = getDimensions().getElementType();
  ArrayRef<StringAttr> symbols = getDimensions().getSymbols();
  ArrayRef<int64_t> integers = getDimensions().getIntegers();
  ArrayRef<bool> shape = getDimensions().getShape();

  printDimensionList(odsPrinter, elemType, symbols, integers, shape);
}

Type ArrayType::generate(GeneratorOpBuilder &builder) {
  llvm::SmallVector<Type> possibleTypes = {
      builder.getIndexType(),
      builder.getI1Type(),
      builder.getI8Type(),
      builder.getI16Type(),
      builder.getI32Type(),
      builder.getI64Type(),
      // builder.getF16Type(),
      builder.getF32Type(),
      builder.getF64Type(),
  };

  if (builder.config.get<unsigned>("sdfg.scientific").value())
    possibleTypes = {builder.getI32Type(), builder.getI64Type(),
                     builder.getF32Type(), builder.getF64Type()};

  Type elemType = builder.sample(possibleTypes).value();
  llvm::SmallVector<int64_t> integers;
  llvm::SmallVector<bool> shape;

  unsigned length = builder.sampleUniform<unsigned>(0, 4);

  if (builder.config.get<unsigned>("sdfg.scientific").value())
    length = builder.sampleUniform<unsigned>(2, 3);

  // Get all dimensions and types of previous arrays
  llvm::SmallVector<Value> prevArrays = builder.collectValues(
      [](const Value &v) { return v.getType().isa<ArrayType>(); });

  llvm::SmallVector<int64_t> prevDimensions;
  llvm::SmallVector<Type> prevTypes;
  for (const Value &v : prevArrays) {
    ArrayType arrayType = v.getType().cast<ArrayType>();
    for (int64_t dim : arrayType.getIntegers())
      prevDimensions.push_back(dim);
    prevTypes.push_back(arrayType.getElementType());
  }

  // In scientific mode reuse previous types 80% of the time
  if (builder.config.get<unsigned>("sdfg.scientific").value() &&
      builder.sampleUniform(0, 100) < 80)
    elemType = builder.sample(prevTypes).value_or(elemType);

  // Generate dimensions
  for (unsigned i = 0; i < length; ++i) {
    int64_t value = builder.sampleGeometric<int64_t>() + 1;
    unsigned limit =
        builder.config
            .get<unsigned>("sdfg.array_dim" + std::to_string(i) + "_limit")
            .value_or(64);

    value = value > limit ? limit : value;

    if (builder.config.get<unsigned>("sdfg.scientific").value()) {
      value = builder.sampleUniform<int64_t>(1, limit);

      if (builder.sampleUniform(0, 100) < 80)
        value = builder.sample(prevDimensions).value_or(value);
    }

    integers.push_back(value);
    shape.push_back(true);
  }

  SizedType sizedType =
      SizedType::get(builder.getContext(), elemType, {}, integers, shape);
  return get(builder.getContext(), sizedType);
}

/// Returns the type of the elements in an array.
Type ArrayType::getElementType() { return getDimensions().getElementType(); }

/// Returns a list of symbols in the array type.
ArrayRef<StringAttr> ArrayType::getSymbols() {
  return getDimensions().getSymbols();
}

/// Returns a list of integer constants in the array type.
ArrayRef<int64_t> ArrayType::getIntegers() {
  return getDimensions().getIntegers();
}

/// Returns a list of booleans representing the shape of the array type.
/// (false = symbolic size, true = integer constant)
ArrayRef<bool> ArrayType::getShape() { return getDimensions().getShape(); }

/// Attempts to parse a stream type.
::mlir::Type StreamType::parse(::mlir::AsmParser &odsParser) {
  Type elementType;
  SmallVector<StringAttr> symbols;
  SmallVector<int64_t> integers;
  SmallVector<bool> shape;
  if (parseDimensionList(odsParser, elementType, symbols, integers, shape))
    return Type();

  SizedType sized = SizedType::get(odsParser.getContext(), elementType, symbols,
                                   integers, shape);
  return get(odsParser.getContext(), sized);
}

/// Prints a stream type in human-readable form.
void StreamType::print(::mlir::AsmPrinter &odsPrinter) const {
  Type elemType = getDimensions().getElementType();
  ArrayRef<StringAttr> symbols = getDimensions().getSymbols();
  ArrayRef<int64_t> integers = getDimensions().getIntegers();
  ArrayRef<bool> shape = getDimensions().getShape();

  printDimensionList(odsPrinter, elemType, symbols, integers, shape);
}

/// Generate the code for type definitions.
#define GET_TYPEDEF_CLASSES
#include "SDFG/Dialect/OpsTypes.cpp.inc"
