#include "SDFG/Translate/Translation.h"
#include "SDFG/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/DenseMap.h"
#include <regex>
#include <string>

using namespace mlir;
using namespace sdfg;
using namespace emitter;
using namespace translation;

//===----------------------------------------------------------------------===//
// Maps for inserting access nodes & creating symbols
//===----------------------------------------------------------------------===//

llvm::DenseMap<Operation *, BlockAndValueMapping> allocMaps;
llvm::DenseMap<Operation *, SmallVector<std::string>> symMaps;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

LogicalResult printRange(Location loc, Attribute &attr, JsonEmitter &jemit) {
  if (StringAttr sym_str = attr.dyn_cast<StringAttr>()) {
    jemit.startObject();
    jemit.printKVPair("start", sym_str.getValue());
    jemit.printKVPair("end", sym_str.getValue());
    jemit.printKVPair("step", 1);
    jemit.printKVPair("tile", 1);
    jemit.endObject();
  } else if (IntegerAttr sym_int = attr.dyn_cast<IntegerAttr>()) {
    jemit.startObject();
    jemit.printKVPair("start", sym_int.getInt());
    jemit.printKVPair("end", sym_int.getInt());
    jemit.printKVPair("step", 1);
    jemit.printKVPair("tile", 1);
    jemit.endObject();
  } else {
    mlir::emitError(loc, "'indices' must consist of StringAttr or IntegerAttr");
    return failure();
  }
  return success();
}

LogicalResult printIndices(Location loc, Attribute attr, JsonEmitter &jemit) {
  if (ArrayAttr syms = attr.dyn_cast<ArrayAttr>()) {
    if (syms.getValue().size() == 0) {
      jemit.startObject();
      jemit.printKVPair("start", 0);
      jemit.printKVPair("end", 0);
      jemit.printKVPair("step", 1);
      jemit.printKVPair("tile", 1);
      jemit.endObject();
    }

    for (Attribute sym : syms.getValue()) {
      if (printRange(loc, sym, jemit).failed())
        return failure();
    }
  } else {
    mlir::emitError(loc, "'indices' must be an ArrayAttr");
    return failure();
  }
  return success();
}

SmallVector<std::string> buildStrideList(ArrayType mem) {
  ArrayRef<bool> shape = mem.getShape();
  ArrayRef<int64_t> integers = mem.getIntegers();
  ArrayRef<StringAttr> symbols = mem.getSymbols();

  SmallVector<std::string> strideList;
  unsigned intIdx = 0;
  unsigned symIdx = 0;

  for (unsigned i = 0; i < shape.size(); ++i) {
    if (shape[i])
      strideList.push_back(std::to_string(integers[intIdx++]));
    else
      strideList.push_back(symbols[symIdx++].str());
  }
  return strideList;
}

SmallVector<std::string> buildStrideList(AllocOp &op) {
  if (ArrayType t = op.getType().dyn_cast<ArrayType>())
    return buildStrideList(t);

  // TODO: Implement super class
  /*   if (StreamType t = op.getType().dyn_cast<StreamType>())
      return buildStrideList(t.toArray()); */

  return SmallVector<std::string>();
}

void printStrides(SmallVector<std::string> strides, JsonEmitter &jemit) {
  for (int i = strides.size() - 1; i > 0; --i) {
    jemit.startEntry();
    jemit.printString(strides[i]);
  }

  jemit.startEntry();
  jemit.printInt(1);
}

std::string getValueName(Value v, Operation &stateOp) {
  std::string name;
  AsmState state(&stateOp);
  llvm::raw_string_ostream nameStream(name);
  v.printAsOperand(nameStream, state);
  utils::sanitizeName(name);
  return name;
}

std::string getTypeName(Type t) {
  std::string name;
  llvm::raw_string_ostream nameStream(name);
  t.print(nameStream);
  return name;
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(ModuleOp &op, JsonEmitter &jemit) {
  utils::resetIDGenerator();

  for (Operation &oper : op.body().getOps())
    if (SDFGNode sdfg = dyn_cast<SDFGNode>(oper))
      if (translateToSDFG(sdfg, jemit).failed())
        return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SDFGNode
//===----------------------------------------------------------------------===//

// TODO: Remove. ConstantOps in SDFGs are obsolete
LogicalResult printConstant(arith::ConstantOp &op, JsonEmitter &jemit) {
  std::string val;
  llvm::raw_string_ostream valStream(val);
  op.getValue().print(valStream);
  val.erase(val.find(' '));

  std::string res = getValueName(op.getResult(), *op);

  jemit.startNamedList(res);

  jemit.startObject();
  jemit.printKVPair("type", "Scalar");

  jemit.startNamedObject("attributes");

  Type type = op.getType();
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(type, loc);

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.startNamedList("shape");
  jemit.printString("1");
  jemit.endList(); // shape
  jemit.printKVPair("transient", "false", /*stringify=*/false);
  jemit.endObject(); // attributes

  jemit.endObject();

  jemit.startEntry();
  jemit.printLiteral(val);
  jemit.endList(); // res

  return success();
}

void translation::prepForTranslation(SDFGNode &op) {
  // Map arguments to arrays
  if (!(*op).hasAttr("arg_names")) {
    BlockAndValueMapping argToAlloc;
    SmallVector<std::string> typeSymbols;

    for (BlockArgument bArg : op.getArguments()) {
      std::string name = getValueName(bArg, *op);

      AllocOp aop = AllocOp::create(op.getLoc(), bArg.getType(), name,
                                    /*transient=*/false);
      bArg.replaceAllUsesExcept(aop, aop);
      op.body().getBlocks().front().push_front(aop);
      argToAlloc.map(bArg, aop);

      if (ArrayType mem = bArg.getType().dyn_cast<ArrayType>())
        for (StringAttr sa : mem.getSymbols())
          typeSymbols.push_back(sa.str());
    }

    allocMaps.insert({op.getOperation(), argToAlloc});
    symMaps.insert({op.getOperation(), typeSymbols});
  }

  for (Operation &oper : op.body().getOps())
    if (StateNode state = dyn_cast<StateNode>(oper))
      prepForTranslation(state);
}

LogicalResult printSDFGNode(SDFGNode &op, JsonEmitter &jemit) {
  jemit.printKVPair("type", "SDFG");
  jemit.printKVPair("sdfg_list_id", utils::generateID(),
                    /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printAttributes(
      op->getAttrs(),
      /*elidedAttrs=*/{"ID", "entry", "sym_name", "type", "arg_names"});
  jemit.printKVPair("name", op.sym_name());

  jemit.startNamedObject("constants_prop");
  // TODO: Remove. Obsolete
  for (StateNode state : op.body().getOps<StateNode>())
    for (arith::ConstantOp constOp : state.body().getOps<arith::ConstantOp>())
      if (printConstant(constOp, jemit).failed())
        return failure();

  jemit.endObject(); // constants_prop

  if ((*op).hasAttr("arg_names")) {
    Attribute arg_names = op->getAttr("arg_names");
    if (ArrayAttr arg_names_arr = arg_names.dyn_cast<ArrayAttr>()) {
      jemit.startNamedList("arg_names");

      for (Attribute arg_name : arg_names_arr.getValue()) {
        if (StringAttr arg_name_str = arg_name.dyn_cast<StringAttr>()) {
          jemit.startEntry();
          jemit.printString(arg_name_str.getValue());
        } else {
          mlir::emitError(op.getLoc(),
                          "'arg_names' must consist of StringAttr");
          return failure();
        }
      }

      jemit.endList(); // arg_names
    } else {
      mlir::emitError(op.getLoc(), "'arg_names' must be an ArrayAttr");
      return failure();
    }
  } else {
    jemit.startNamedList("arg_names");
    for (BlockArgument bArg : op.getArguments()) {
      std::string name = getValueName(bArg, *op);
      jemit.startEntry();
      jemit.printString(name);
    }
    jemit.endList(); // arg_names
  }

  jemit.startNamedObject("_arrays");

  for (AllocOp alloc : op.body().getOps<AllocOp>())
    if (translateToSDFG(alloc, jemit).failed())
      return failure();

  for (StateNode state : op.body().getOps<StateNode>()) {
    for (AllocOp allocOper : state.body().getOps<AllocOp>())
      if (translateToSDFG(allocOper, jemit).failed())
        return failure();
  }

  jemit.endObject(); // _arrays

  jemit.startNamedObject("symbols");
  for (std::string s : symMaps.lookup(op)) {
    AllocSymbolOp aso = AllocSymbolOp::create(op.getLoc(), s);
    if (translateToSDFG(aso, jemit).failed())
      return failure();
  }

  for (Operation &oper : op.body().getOps())
    if (AllocSymbolOp alloc = dyn_cast<AllocSymbolOp>(oper))
      if (translateToSDFG(alloc, jemit).failed())
        return failure();

  jemit.endObject(); // symbols
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  unsigned stateID = 0;

  for (Operation &oper : op.body().getOps())
    if (StateNode state = dyn_cast<StateNode>(oper)) {
      state.setID(stateID);
      if (translateToSDFG(state, jemit).failed())
        return failure();
      stateID++;
    }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");

  for (Operation &oper : op.body().getOps())
    if (EdgeOp edge = dyn_cast<EdgeOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();

  jemit.endList(); // edges

  StateNode entryState = op.getStateBySymRef(op.entry());
  jemit.printKVPair("start_state", entryState.ID(), /*stringify=*/false);

  return success();
}

LogicalResult translation::translateToSDFG(SDFGNode &op, JsonEmitter &jemit) {
  prepForTranslation(op);

  if (!op.isNested()) {
    jemit.startObject();

    if (printSDFGNode(op, jemit).failed())
      return failure();

    jemit.endObject();
    return success();
  }

  jemit.startObject();
  jemit.printKVPair("type", "NestedSDFG");
  jemit.printKVPair("id", op.ID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", op.sym_name());

  jemit.startNamedObject("symbol_mapping");

  for (std::string mappedSy : symMaps.lookup(op))
    jemit.printKVPair(mappedSy, mappedSy);

  jemit.endObject(); // symbol_mapping

  jemit.startNamedObject("in_connectors");
  for (BlockArgument bArg : op.getArguments()) {
    std::string name = getValueName(bArg, *op);
    jemit.printKVPair(name, "null", /*stringify=*/false);
  }
  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  // TODO: Implement multiple return values
  // Takes the form __return_%d
  if (op.getNumResults() == 1) {
    jemit.printKVPair("__return", "null", /*stringify=*/false);
  } else if (op.getNumResults() > 1) {
    emitError(op.getLoc(), "Multiple return values not implemented yet");
    return failure();
  }

  jemit.endObject(); // out_connectors

  jemit.startNamedObject("sdfg");

  if (printSDFGNode(op, jemit).failed())
    return failure();

  jemit.endObject(); // sdfg
  jemit.endObject(); // attributes

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// StateNode
//===----------------------------------------------------------------------===//

void translation::prepForTranslation(StateNode &op) {
  // Remove dead loads
  if (op.body().getBlocks().front().getOperations().size() > 0 &&
      isa<LoadOp>(op.body().getBlocks().front().back()))
    op.body().getBlocks().front().back().erase();

  // separate operations requiring indirects
  SmallVector<LoadOp> indirectsLoads;
  SmallVector<StoreOp> indirectsStores;
  for (Operation &oper : op.body().getOps()) {
    if (StoreOp edge = dyn_cast<StoreOp>(oper))
      if (edge.isIndirect())
        indirectsStores.push_back(edge);

    if (LoadOp edge = dyn_cast<LoadOp>(oper))
      if (edge.isIndirect())
        indirectsLoads.push_back(edge);
  }

  // Rewrite indirect operations
  for (LoadOp load : indirectsLoads) {
    FunctionType ft = FunctionType::get(op.getContext(), load.getOperandTypes(),
                                        load.getType());

    TaskletNode task = TaskletNode::create(op.getLoc(), "indirect_load", ft);

    BlockAndValueMapping valMapping;
    valMapping.map(load.getOperands(), task.getArguments());

    Operation *copy = load.getOperation()->clone(valMapping);
    task.body().getBlocks().front().push_back(copy);

    ReturnOp ret = ReturnOp::create(op.getLoc(), copy->getResults());
    task.body().getBlocks().front().push_back(ret);
    op.body().getBlocks().front().push_front(task);

    CallOp call = CallOp::create(op.getLoc(), task, load.getOperands());
    OpBuilder builder(op.getLoc().getContext());
    builder.setInsertionPointAfter(load);
    builder.insert(call);

    load.replaceAllUsesWith(call);
    load.erase();
  }

  for (StoreOp store : indirectsStores) {
    OpBuilder builder(op.getLoc().getContext());
    builder.setInsertionPoint(store);

    Type t = store.arr().getType();
    if (!t.isa<ArrayType>())
      t = ArrayType::get(op.getLoc().getContext(), t, {}, {}, {});

    SmallVector<Value> reducedOps;
    SmallVector<Type> reducedTypes;

    for (unsigned i = 0; i < store.getNumOperands() - 1; ++i) {
      reducedOps.push_back(store.getOperand(i));
      reducedTypes.push_back(store.getOperand(i).getType());
    }

    FunctionType ft = FunctionType::get(op.getContext(), reducedTypes, t);

    TaskletNode task = TaskletNode::create(op.getLoc(), "indirect_store", ft);

    AllocOp aop = cast<AllocOp>(store.arr().getDefiningOp());
    BlockAndValueMapping valMapping;
    valMapping.map(reducedOps, task.getArguments());
    valMapping.map(store.arr(), aop);

    Operation *copy = store.getOperation()->clone(valMapping);
    task.body().getBlocks().front().push_back(copy);

    ReturnOp ret = ReturnOp::create(op.getLoc(), copy->getResults());
    task.body().getBlocks().front().push_back(ret);
    builder.insert(task);

    CallOp call = CallOp::create(op.getLoc(), task, reducedOps);
    builder.insert(call);

    StoreOp newStore = StoreOp::create(op.getLoc(), call.getResult(0), aop);
    builder.insert(newStore);

    store.erase();
  }

  // Wrap symbolic evaluations
  SmallVector<SymOp> deadSymbols;
  for (Operation &oper : op.body().getOps()) {
    if (SymOp sym = dyn_cast<SymOp>(oper)) {
      deadSymbols.push_back(sym);
      if (sym.use_empty())
        continue;

      FunctionType ft = FunctionType::get(op.getContext(), {}, sym.getType());
      TaskletNode task = TaskletNode::create(op.getLoc(), "sym_task", ft);

      Operation *copy = sym->clone();
      task.body().getBlocks().front().push_back(copy);

      ReturnOp ret = ReturnOp::create(op.getLoc(), copy->getResults());
      task.body().getBlocks().front().push_back(ret);
      op.body().getBlocks().front().push_front(task);

      CallOp call = CallOp::create(op.getLoc(), task, {});

      OpBuilder builder(op.getLoc().getContext());
      builder.setInsertionPointAfter(sym);

      Type t = sym.getType();
      if (!t.isa<ArrayType>())
        t = ArrayType::get(op.getLoc().getContext(), t, {}, {}, {});

      AllocOp aop =
          AllocOp::create(op.getLoc(), t, utils::generateName("sym_wrap"),
                          /*transient=*/true);

      StoreOp store =
          StoreOp::create(op.getLoc(), call.getResult(0), aop, ValueRange());
      LoadOp load = LoadOp::create(op.getLoc(), aop, {});

      builder.insert(aop);
      builder.insert(call);
      builder.insert(store);
      builder.insert(load);
      sym.replaceAllUsesWith(load.res());
    }
  }

  for (SymOp symOp : deadSymbols)
    symOp.erase();

  for (Operation &oper : op.body().getOps()) {
    if (CallOp call = dyn_cast<CallOp>(oper))
      prepForTranslation(call);
  }
}

LogicalResult translation::translateToSDFG(StateNode &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  jemit.printKVPair("label", op.sym_name());
  jemit.printKVPair("id", op.ID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.printAttributes(op->getAttrs(), /*elidedAttrs=*/{"ID", "sym_name"});
  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");

  unsigned nodeID = 0;
  for (Operation &oper : op.body().getOps()) {
    if (TaskletNode tasklet = dyn_cast<TaskletNode>(oper)) {
      tasklet.setID(nodeID++);
      if (translateToSDFG(tasklet, jemit).failed())
        return failure();
    }

    if (SDFGNode sdfg = dyn_cast<SDFGNode>(oper)) {
      sdfg.setID(nodeID++);
      if (translateToSDFG(sdfg, jemit).failed())
        return failure();
    }

    if (MapNode map = dyn_cast<MapNode>(oper)) {
      map.setEntryID(nodeID++);
      map.setExitID(nodeID++);
      if (translateToSDFG(map, jemit).failed())
        return failure();
    }

    if (ConsumeNode consume = dyn_cast<ConsumeNode>(oper)) {
      consume.setEntryID(nodeID++);
      consume.setExitID(nodeID++);
      if (translateToSDFG(consume, jemit).failed())
        return failure();
    }
  }

  jemit.endList(); // nodes

  jemit.startNamedList("edges");

  for (Operation &oper : op.body().getOps()) {
    if (StoreOp edge = dyn_cast<StoreOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();

    if (CopyOp edge = dyn_cast<CopyOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();

    if (sdfg::CallOp edge = dyn_cast<sdfg::CallOp>(oper))
      if (translateToSDFG(edge, jemit).failed())
        return failure();
  }

  jemit.endList(); // edges

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// TaskletNode
//===----------------------------------------------------------------------===//

// TODO(later): Temporary auto-lifting. Will be included into DaCe
LogicalResult liftToPython(TaskletNode &op, JsonEmitter &jemit) {
  int numOps = 0;
  Operation *firstOp = nullptr;

  for (Operation &oper : op.body().getOps()) {
    if (numOps == 0)
      firstOp = &oper;
    ++numOps;
  }

  if (numOps > 2) {
    emitRemark(op.getLoc(), "No lifting to python possible");
    return failure();
  }

  if (isa<arith::AddFOp>(firstOp) || isa<arith::AddIOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    std::string nameArg1 = op.getInputName(op.getNumArguments() - 1);

    jemit.printKVPair("string_data", "__out = " + nameArg0 + " + " + nameArg1);
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (isa<arith::MulFOp>(firstOp) || isa<arith::MulIOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    std::string nameArg1 = op.getInputName(op.getNumArguments() - 1);

    jemit.printKVPair("string_data", "__out = " + nameArg0 + " * " + nameArg1);
    jemit.printKVPair("language", "Python");
    return success();
  }

  // TODO: Add arith ops

  if (arith::ConstantOp oper = dyn_cast<arith::ConstantOp>(firstOp)) {
    std::string val;

    if (arith::ConstantFloatOp flop =
            dyn_cast<arith::ConstantFloatOp>(firstOp)) {
      SmallVector<char> flopVec;
      flop.value().toString(flopVec);
      for (char c : flopVec)
        val += c;
    } else if (arith::ConstantIntOp iop =
                   dyn_cast<arith::ConstantIntOp>(firstOp)) {
      val = std::to_string(iop.value());
    } else if (arith::ConstantIndexOp iop =
                   dyn_cast<arith::ConstantIndexOp>(firstOp)) {
      val = std::to_string(iop.value());
    }

    std::string entry = "__out = " + val;
    jemit.printKVPair("string_data", entry);
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (arith::IndexCastOp ico = dyn_cast<arith::IndexCastOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    std::string entry = "__out = " + nameArg0;
    jemit.printKVPair("string_data", entry);
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (StoreOp store = dyn_cast<StoreOp>(firstOp)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumArguments() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(op.getInputName(i));
    }

    std::string valName = op.getInputName(op.getNumArguments() - 1);

    jemit.printKVPair("string_data",
                      "__out[" + indices + "]" + " = " + valName);
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (isa<LoadOp>(firstOp)) {
    std::string indices;

    for (unsigned i = 0; i < op.getNumArguments() - 1; ++i) {
      if (i > 0)
        indices.append(", ");
      indices.append(op.getInputName(i));
    }

    std::string arrName = op.getInputName(op.getNumArguments() - 1);

    jemit.printKVPair("string_data",
                      "__out = " + arrName + "[" + indices + "]");
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (SymOp sym = dyn_cast<SymOp>(firstOp)) {
    jemit.printKVPair("string_data", "__out = " + sym.expr().str());
    jemit.printKVPair("language", "Python");
    return success();
  }

  if (isa<sdfg::ReturnOp>(firstOp)) {
    std::string nameArg0 = op.getInputName(0);
    jemit.printKVPair("string_data", "__out = " + nameArg0);
    jemit.printKVPair("language", "Python");
    return success();
  }

  emitRemark(op.getLoc(), "No lifting to python possible");
  return failure();
}

LogicalResult translation::translateToSDFG(TaskletNode &op,
                                           JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Tasklet");
  jemit.printKVPair("label", op.sym_name());

  jemit.startNamedObject("attributes");
  jemit.printKVPair("label", op.sym_name());

  jemit.startNamedObject("code");

  // Try to lift the body of the tasklet
  // If lifting fails (body is complex) then emit MLIR code directly
  // liftToPython() emits automatically emits the generated python code
  if (liftToPython(op, jemit).failed()) {
    // Convention: MLIR tasklets use the mlir_entry function as the entry
    // point
    std::string code = "module {\\n func @mlir_entry(";

    // Prints all arguments with types
    for (unsigned i = 0; i < op.getNumArguments(); ++i) {
      BlockArgument bArg = op.getArgument(i);
      std::string name = getValueName(bArg, *op);
      std::string type = getTypeName(bArg.getType());

      if (i > 0)
        code.append(", ");
      code.append(name);
      code.append(": ");
      code.append(type);
    }

    code.append(") -> ");

    for (Type res : op.getCallableResults())
      code.append(getTypeName(res));

    code.append(" {\\n");

    // Emits the body of the tasklet
    for (Operation &oper : op.body().getOps()) {
      std::string codeLine;
      llvm::raw_string_ostream codeLineStream(codeLine);
      oper.print(codeLineStream);

      // SDFG is not a core dialect. Therefore "sdfg.return" does not exist
      // Replace it with the standard "return"
      if (sdfg::ReturnOp ret = dyn_cast<sdfg::ReturnOp>(oper))
        codeLine.replace(codeLine.find("sdfg.return"), 11, "return");

      codeLine.append("\\n");
      code.append(codeLine);
    }

    code.append("}\\n}");

    std::size_t n = code.length();
    std::string escapedCode;

    for (std::size_t i = 0; i < n; ++i) {
      if (code[i] == '\\' || code[i] == '\"')
        escapedCode += '\\';
      escapedCode += code[i];
    }
    jemit.printKVPair("string_data", escapedCode);
    jemit.printKVPair("language", "MLIR");
  }

  jemit.endObject(); // code

  jemit.startNamedObject("in_connectors");

  for (unsigned i = 0; i < op.getNumArguments(); ++i) {
    jemit.printKVPair(op.getInputName(i), "null",
                      /*stringify=*/false);
  }

  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  // TODO: Implement multiple return values
  // Takes the form __out_%d
  if (op.getNumResults() == 1) {
    jemit.printKVPair(op.getOutputName(0), "null", /*stringify=*/false);
  } else if (op.getNumResults() > 1) {
    emitError(op.getLoc(), "Multiple return values not implemented yet");
    return failure();
  }
  jemit.endObject(); // out_connectors

  jemit.endObject(); // attributes

  jemit.printKVPair("id", op.ID(), /*stringify=*/false);
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// MapNode
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(MapNode &op, JsonEmitter &jemit) {
  // MapEntry
  jemit.startObject();
  jemit.printKVPair("type", "MapEntry");

  jemit.startNamedObject("attributes");

  jemit.startNamedList("params");
  for (BlockArgument arg : op.getBody()->getArguments()) {
    jemit.startEntry();
    std::string name = getValueName(arg, *op);
    jemit.printString(name);
  }
  jemit.endList(); // params

  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.entryID(), /*stringify=*/false);
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();

  // MapExit
  jemit.startObject();
  jemit.printKVPair("type", "MapExit");

  jemit.startNamedObject("attributes");

  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.exitID(), /*stringify=*/false);
  jemit.printKVPair("scope_entry", op.entryID());
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// ConsumeNode
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(ConsumeNode &op,
                                           JsonEmitter &jemit) {
  // ConsumeEntry
  jemit.startObject();
  jemit.printKVPair("type", "ConsumeEntry");

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.entryID(), /*stringify=*/false);
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();

  // ConsumeExit
  jemit.startObject();
  jemit.printKVPair("type", "ConsumeExit");

  jemit.startNamedObject("attributes");

  jemit.endObject(); // attributes
  jemit.printKVPair("id", op.exitID(), /*stringify=*/false);
  jemit.printKVPair("scope_entry", op.entryID());
  jemit.printKVPair("scope_exit", op.exitID());

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// EdgeOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(EdgeOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Edge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "InterstateEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("assignments");

  std::string refname = "";
  if (!op.refMutable().empty()) {
    if (AllocOp aop = dyn_cast<AllocOp>(op.ref().getDefiningOp())) {
      refname = aop.getName();
    }
  }

  if (op.assign().hasValue()) {
    ArrayAttr assignments = op.assign().getValue();

    for (Attribute assignment : assignments) {
      if (StringAttr strAttr = assignment.dyn_cast<StringAttr>()) {
        StringRef content = strAttr.getValue();
        std::pair<StringRef, StringRef> kv = content.split(':');
        std::string replaced = std::regex_replace(kv.second.trim().str(),
                                                  std::regex("ref"), refname);
        jemit.printKVPair(kv.first.trim(), replaced);
      } else {
        mlir::emitError(
            op.getLoc(),
            "'assign' must be an ArrayAttr consisting of StringAttr");
        return failure();
      }
    }
  }

  jemit.endObject(); // assignments
  jemit.startNamedObject("condition");
  jemit.printKVPair("language", "Python");
  if (op.condition().hasValue()) {
    if (op.condition().getValue().empty()) {
      jemit.printKVPair("string_data", "1");
    } else {
      std::string cond = op.condition().getValue().trim().str();
      std::string replaced =
          std::regex_replace(cond, std::regex("ref"), refname);
      jemit.printKVPair("string_data", replaced);
    }
  } else {
    jemit.printKVPair("string_data", "1");
  }
  jemit.endObject(); // condition
  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  SDFGNode sdfg = dyn_cast<SDFGNode>(op->getParentOp());

  StateNode srcState = sdfg.getStateBySymRef(op.src());
  unsigned srcIdx = sdfg.getIndexOfState(srcState);
  jemit.printKVPair("src", srcIdx);

  StateNode destState = sdfg.getStateBySymRef(op.dest());
  unsigned destIdx = sdfg.getIndexOfState(destState);
  jemit.printKVPair("dst", destIdx);

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(AllocOp &op, JsonEmitter &jemit) {
  jemit.startNamedObject(op.getName());
  jemit.printKVPair("type", "Array");

  jemit.startNamedObject("attributes");

  Type element = op.getElementType();
  Location loc = op.getLoc();
  StringRef dtype = translateTypeToSDFG(element, loc);
  SmallVector<std::string> strideList;

  if (!op.isScalar()) {
    jemit.startNamedList("strides");
    strideList = buildStrideList(op);
    printStrides(strideList, jemit);
    jemit.endList(); // strides
  }

  if (dtype != "")
    jemit.printKVPair("dtype", dtype);
  else
    return failure();

  jemit.startNamedList("shape");
  for (std::string s : strideList) {
    jemit.startEntry();
    jemit.printString(s);
  }

  if (op.isScalar()) {
    jemit.startEntry();
    jemit.printInt(1);
  }
  jemit.endList(); // shape

  jemit.printKVPair("transient", op.transient() ? "true" : "false",
                    /*stringify=*/false);
  printDebuginfo(*op, jemit);

  jemit.endObject(); // attributes
  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// GetAccessOp
//===----------------------------------------------------------------------===//

// LogicalResult translation::translateToSDFG(GetAccessOp &op,
//                                            JsonEmitter &jemit) {
/*   jemit.startObject();
  jemit.printKVPair("type", "AccessNode");
  jemit.printKVPair("label", op.getName());

  jemit.startNamedObject("attributes");
  jemit.printKVPair("data", op.getName());
  jemit.startNamedObject("in_connectors");
  jemit.endObject(); // in_connectors
  jemit.startNamedObject("out_connectors");
  jemit.endObject(); // out_connectors
  jemit.endObject(); // attributes */

// jemit.printKVPair("id", op.ID(), /*stringify=*/false);
// jemit.printKVPair("scope_entry", "null", /*stringify=*/false);
// jemit.printKVPair("scope_exit", "null", /*stringify=*/false);
/*   jemit.endObject();
  return success(); */
//}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(LoadOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateLoadToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StoreOp &op, JsonEmitter &jemit) {
  bool fullRange = op->getAttr("isFullRange") != nullptr &&
                   op->getAttr("isFullRange").cast<BoolAttr>().getValue();

  AllocOp aop = cast<AllocOp>(op.arr().getDefiningOp());

  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  jemit.printKVPair("data", aop.getName());
  jemit.printKVPair("volume", 1);

  if (!fullRange) {
    jemit.startNamedList("strides");
    SmallVector<std::string> strideList = buildStrideList(aop);
    printStrides(strideList, jemit);
    jemit.endList(); // strides

    jemit.startNamedObject("subset");
    jemit.printKVPair("type", "Range");
    jemit.startNamedList("ranges");
    if (printIndices(op.getLoc(), op->getAttr("indices"), jemit).failed())
      return failure();
    jemit.endList();   // ranges
    jemit.endObject(); // subset

    jemit.startNamedObject("dst_subset");
    jemit.printKVPair("type", "Range");
    jemit.startNamedList("ranges");

    if (printIndices(op.getLoc(), op->getAttr("indices"), jemit).failed())
      return failure();

    jemit.endList();   // ranges
    jemit.endObject(); // dst_subset
  }

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  // Get the ID of the tasklet/SDFG if this StoreOp represents
  // a tasklet/nested SDFG -> access node edge
  if (sdfg::CallOp call = dyn_cast<sdfg::CallOp>(op.val().getDefiningOp())) {
    if (call.callsTasklet()) {
      TaskletNode aNode = call.getTasklet();
      jemit.printKVPair("src", aNode.ID());
      // TODO: Implement multiple return values
      // Takes the form __out_%d
      jemit.printKVPair("src_connector", aNode.getOutputName(0));
    } else {
      SDFGNode aNode = call.getSDFG();
      jemit.printKVPair("src", aNode.ID());
      // TODO: Implement multiple return values
      // Takes the form __return_%d
      jemit.printKVPair("src_connector", "__return");
    }
  } else if (LoadOp load = dyn_cast<LoadOp>(op.val().getDefiningOp())) {
    AllocOp aop = cast<AllocOp>(load.arr().getDefiningOp());
    // TODO: Emit ID
    // jemit.printKVPair("src", aop.ID());
    jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(op.getLoc(),
                    "Value must be result of TaskletNode or LoadOp");
    return failure();
  }

  if (AllocOp aNode = dyn_cast<AllocOp>(op.arr().getDefiningOp())) {
    // TODO: Emit ID
    // jemit.printKVPair("dst", aNode.ID());
    jemit.printKVPair("dst_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(op.getLoc(), "Array must be defined by an AllocOp");
    return failure();
  }

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(CopyOp &op, JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  if (AllocOp aNode = dyn_cast<AllocOp>(op.src().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
    jemit.printKVPair("volume", 1);
  } else {
    mlir::emitError(op.getLoc(), "Source array must be defined by an AllocOp");
    return failure();
  }

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  if (AllocOp aNode = dyn_cast<AllocOp>(op.src().getDefiningOp())) {
    // TODO: Emit ID
    // jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(op.getLoc(), "Source array must be defined by an AllocOp");
    return failure();
  }

  if (AllocOp aNode = dyn_cast<AllocOp>(op.dest().getDefiningOp())) {
    // TODO: Emit ID
    // jemit.printKVPair("dst", aNode.ID());
    jemit.printKVPair("dst_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(op.getLoc(),
                    "Destination array must be defined by an AllocOp");
    return failure();
  }

  jemit.endObject();
  return success();
}

//===----------------------------------------------------------------------===//
// MemletCastOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(MemletCastOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateMemletCastToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// ViewCastOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(ViewCastOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateViewCastToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(SubviewOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateSubviewToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// StreamPopOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StreamPopOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateStreamPopToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// StreamPushOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StreamPushOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateStreamPushToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// StreamLengthOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(StreamLengthOp &op,
                                           JsonEmitter &jemit) {
  // TODO: Implement translateStreamLengthToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult printLoadEdgeAttr(LoadOp &load, JsonEmitter &jemit) {
  jemit.startNamedList("strides");
  AllocOp aop = cast<AllocOp>(load.arr().getDefiningOp());
  SmallVector<std::string> strideList = buildStrideList(aop);
  printStrides(strideList, jemit);
  jemit.endList(); // strides

  if (AllocOp aNode = dyn_cast<AllocOp>(load.arr().getDefiningOp())) {
    jemit.printKVPair("data", aNode.getName());
    jemit.printKVPair("volume", 1);
  } else {
    mlir::emitError(load.getLoc(), "Array must be defined by an AllocOp");
    return failure();
  }

  jemit.startNamedObject("subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");
  if (printIndices(load.getLoc(), load->getAttr("indices"), jemit).failed())
    return failure();
  jemit.endList();   // ranges
  jemit.endObject(); // subset

  jemit.startNamedObject("src_subset");
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");
  if (printIndices(load.getLoc(), load->getAttr("indices"), jemit).failed())
    return failure();
  jemit.endList();   // ranges
  jemit.endObject(); // src_subset
  return success();
}

void printMultiConnectorStart(JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");
  jemit.startNamedObject("attributes");
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");
}

void printMultiConnectorAttrEnd(JsonEmitter &jemit) {
  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes
}

void printMultiConnectorEnd(JsonEmitter &jemit) { jemit.endObject(); }

LogicalResult printLoadTaskletEdge(LoadOp &load, TaskletNode &task, int argIdx,
                                   JsonEmitter &jemit) {
  printMultiConnectorStart(jemit);
  if (printLoadEdgeAttr(load, jemit).failed())
    return failure();
  printMultiConnectorAttrEnd(jemit);

  if (AllocOp aNode = dyn_cast<AllocOp>(load.arr().getDefiningOp())) {
    // TODO: Emit ID
    // jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(load.getLoc(), "Array must be defined by an AllocOp");
    return failure();
  }

  jemit.printKVPair("dst", task.ID());
  jemit.printKVPair("dst_connector", task.getInputName(argIdx));
  printMultiConnectorEnd(jemit);
  return success();
}

LogicalResult printTaskletTaskletEdge(TaskletNode &taskSrc,
                                      TaskletNode &taskDest, int argIdx,
                                      JsonEmitter &jemit) {
  printMultiConnectorStart(jemit);
  printMultiConnectorAttrEnd(jemit);

  jemit.printKVPair("src", taskSrc.ID());
  // TODO: Implement multiple return values
  // Takes the form __out_%d
  jemit.printKVPair("src_connector", taskSrc.getOutputName(0));

  jemit.printKVPair("dst", taskDest.ID());
  jemit.printKVPair("dst_connector", taskDest.getInputName(argIdx));

  printMultiConnectorEnd(jemit);
  return success();
}

/* LogicalResult printAccessTaskletEdge(GetAccessOp &access, TaskletNode &task,
                                     int argIdx, JsonEmitter &jemit) {
  printMultiConnectorStart(jemit);
  jemit.printKVPair("data", access.getName());
  jemit.printKVPair("volume", 1);
  printMultiConnectorAttrEnd(jemit);

  jemit.printKVPair("src", access.ID());
  jemit.printKVPair("src_connector", "null", false);
  jemit.printKVPair("dst", task.ID());
  jemit.printKVPair("dst_connector", task.getInputName(argIdx));
  printMultiConnectorEnd(jemit);
  return success();
} */

/* LogicalResult printAccessSDFGEdge(GetAccessOp &access, SDFGNode &sdfg,
                                  int argIdx, JsonEmitter &jemit) {
  printMultiConnectorStart(jemit);
  jemit.printKVPair("data", access.getName());
  jemit.printKVPair("volume", 1);
  printMultiConnectorAttrEnd(jemit);

  jemit.printKVPair("src", access.ID());
  jemit.printKVPair("src_connector", "null", false);
  jemit.printKVPair("dst", sdfg.ID());
  jemit.printKVPair("dst_connector",
                    getValueName(sdfg.getArgument(argIdx), *sdfg));

  printMultiConnectorEnd(jemit);
  return success();
} */

LogicalResult printTaskletSDFGEdge(TaskletNode &task, SDFGNode &sdfg,
                                   int argIdx, JsonEmitter &jemit) {
  printMultiConnectorStart(jemit);
  printMultiConnectorAttrEnd(jemit);

  jemit.printKVPair("src", task.ID());
  // TODO: Implement multiple return values
  // Takes the form __out_%d
  jemit.printKVPair("src_connector", task.getOutputName(0));
  jemit.printKVPair("dst", sdfg.ID());
  jemit.printKVPair("dst_connector",
                    getValueName(sdfg.getArgument(argIdx), *sdfg));

  printMultiConnectorEnd(jemit);
  return success();
}

LogicalResult printLoadSDFGEdge(LoadOp &load, SDFGNode &sdfg, int argIdx,
                                JsonEmitter &jemit) {
  printMultiConnectorStart(jemit);
  if (printLoadEdgeAttr(load, jemit).failed())
    return failure();
  printMultiConnectorAttrEnd(jemit);

  if (AllocOp aNode = dyn_cast<AllocOp>(load.arr().getDefiningOp())) {
    // TODO: Emit ID
    // jemit.printKVPair("src", aNode.ID());
    jemit.printKVPair("src_connector", "null", /*stringify=*/false);
  } else {
    mlir::emitError(load.getLoc(), "Array must be defined by an AllocOp");
    return failure();
  }

  jemit.printKVPair("dst", sdfg.ID());
  jemit.printKVPair("dst_connector",
                    getValueName(sdfg.getArgument(argIdx), *sdfg));

  printMultiConnectorEnd(jemit);
  return success();
}

void translation::prepForTranslation(sdfg::CallOp &op) {
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    Value val = op.getOperand(i);

    if (sdfg::CallOp call = dyn_cast<sdfg::CallOp>(val.getDefiningOp())) {
      OpBuilder builder(op.getLoc().getContext());
      FunctionType ft = builder.getFunctionType(call.getOperandTypes(),
                                                call.getResultTypes());

      // TODO: Only one output supported
      Type t = ft.getResult(0);
      if (!t.isa<ArrayType>()) {
        t = ArrayType::get(op.getLoc().getContext(), t, {}, {}, {});
      }

      AllocOp alloc =
          AllocOp::create(op.getLoc(), t, utils::generateName("ttt"),
                          /*transient=*/true);
      StoreOp store =
          StoreOp::create(op.getLoc(), call.getResult(0), alloc, ValueRange());
      LoadOp load = LoadOp::create(op.getLoc(), alloc, {});

      builder.setInsertionPointAfter(call);
      builder.insert(alloc);
      ;
      builder.insert(store);
      builder.insert(load);

      val.replaceAllUsesExcept(load, store);
    }
  }
}

LogicalResult translation::translateToSDFG(sdfg::CallOp &op,
                                           JsonEmitter &jemit) {
  // TODO: This can be refactored to avoid code duplication
  if (op.callsTasklet()) {
    TaskletNode task = op.getTasklet();
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value val = op.getOperand(i);
      if (LoadOp load = dyn_cast<LoadOp>(val.getDefiningOp())) {
        if (printLoadTaskletEdge(load, task, i, jemit).failed())
          return failure();

      } else if (sdfg::CallOp call =
                     dyn_cast<sdfg::CallOp>(val.getDefiningOp())) {
        TaskletNode taskSrc = call.getTasklet();
        if (printTaskletTaskletEdge(taskSrc, task, i, jemit).failed())
          return failure();
      } /* else if (GetAccessOp acc =
      dyn_cast<GetAccessOp>(val.getDefiningOp())) { if
      (printAccessTaskletEdge(acc, task, i, jemit).failed()) return failure();
      }  */
      else {
        mlir::emitError(op.getLoc(), "Operands must be results of GetAccessOp, "
                                     "LoadOp or TaskletNode");
        return failure();
      }
    }
  } else {
    // calls nested SDFG
    SDFGNode sdfg = op.getSDFG();
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value val = op.getOperand(i);
      /*       if (GetAccessOp acc = dyn_cast<GetAccessOp>(val.getDefiningOp()))
         { if (printAccessSDFGEdge(acc, sdfg, i, jemit).failed()) return
         failure(); } else */
      if (sdfg::CallOp call = dyn_cast<sdfg::CallOp>(val.getDefiningOp())) {
        TaskletNode taskSrc = call.getTasklet();
        if (printTaskletSDFGEdge(taskSrc, sdfg, i, jemit).failed())
          return failure();
      } else if (LoadOp load = dyn_cast<LoadOp>(val.getDefiningOp())) {
        if (printLoadSDFGEdge(load, sdfg, i, jemit).failed())
          return failure();

      } else {
        mlir::emitError(
            op.getLoc(),
            "Operands must be results of GetAccessOp, LoadOp or TaskletNode");
        return failure();
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LibCallOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(LibCallOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateLibCallToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// AllocSymbolOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(AllocSymbolOp &op,
                                           JsonEmitter &jemit) {
  jemit.printKVPair(op.sym(), "int64");
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolExprOp
//===----------------------------------------------------------------------===//

LogicalResult translation::translateToSDFG(SymOp &op, JsonEmitter &jemit) {
  // TODO: Implement translateSymbolExprToSDFG
  return success();
}

//===----------------------------------------------------------------------===//
// Translate type
//===----------------------------------------------------------------------===//

StringRef translation::translateTypeToSDFG(Type &t, Location &loc) {
  if (t.isF64())
    return "float64";

  if (t.isF32())
    return "float32";

  if (t.isInteger(64))
    return "int64";

  if (t.isInteger(32))
    return "int32";

  if (t.isIndex())
    return "int64";

  std::string type = getTypeName(t);
  mlir::emitError(loc, "Unsupported type: " + type);

  return "";
}

//===----------------------------------------------------------------------===//
// Print debuginfo
//===----------------------------------------------------------------------===//

inline void translation::printDebuginfo(Operation &op, JsonEmitter &jemit) {
  /*std::string loc;
  llvm::raw_string_ostream locStream(loc);
  op.getLoc().print(locStream);
  remove(loc.begin(), loc.end(), '\"');
  jemit.printKVPair("debuginfo", loc);*/

  /*jemit.startNamedObject("debuginfo");
  jemit.printKVPair("type", "DebugInfo");
  jemit.printKVPair("start_line", 1);
  jemit.printKVPair("end_line", 1);
  jemit.printKVPair("start_column", 1);
  jemit.printKVPair("end_column", 1);
  jemit.printKVPair("filename", 1);

  jemit.endObject(); // debuginfo*/
}