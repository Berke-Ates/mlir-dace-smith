// Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

/// This file contains the nodes of the internal IR used by the translator.

#include "SDFG/Translate/Node.h"

using namespace mlir;
using namespace sdfg;
using namespace translation;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Converts a MLIR type to a DaCe DType.
DType typeToDtype(Type t) {
  if (t.isInteger(1))
    return DType::boolean;

  if (t.isInteger(8))
    return DType::int8;

  if (t.isInteger(16))
    return DType::int16;

  if (t.isInteger(32))
    return DType::int32;

  if (t.isInteger(64))
    return DType::int64;

  if (t.isF16())
    return DType::float16;

  if (t.isF32())
    return DType::float32;

  if (t.isF64())
    return DType::float64;

  if (t.isIndex())
    return DType::int64;

  std::string type;
  llvm::raw_string_ostream typeStream(type);
  t.print(typeStream);

  emitWarning(Builder(t.getContext()).getUnknownLoc(),
              "Unsupported Type: " + type);

  return DType::null;
}

/// Converts a DType to a string.
std::string dtypeToString(DType t) {
  switch (t) {
  case DType::boolean:
    return "bool";
  case DType::int8:
    return "int8";
  case DType::int16:
    return "int16";
  case DType::int32:
    return "int32";
  case DType::int64:
    return "int64";
  case DType::float16:
    return "float16";
  case DType::float32:
    return "float32";
  case DType::float64:
    return "float64";
  case DType::null:
    return "null";
  }

  return "Unsupported DType";
}

/// Converts a CodeLanguage to a string.
std::string codeLanguageToString(CodeLanguage lang) {
  switch (lang) {
  case CodeLanguage::Python:
    return "Python";
  case CodeLanguage::CPP:
    return "CPP";
  case CodeLanguage::MLIR:
    return "MLIR";
  }

  return "Unsupported CodeLanguage";
}

/// Prints an array of ranges to the output stream.
void printRangeVector(std::vector<translation::Range> ranges, std::string name,
                      emitter::JsonEmitter &jemit) {
  if (ranges.empty()) {
    jemit.printKVPair(name, "null", /*stringify=*/false);
    return;
  }

  jemit.startNamedObject(name);
  jemit.printKVPair("type", "Range");
  jemit.startNamedList("ranges");
  for (translation::Range r : ranges)
    r.emit(jemit);
  jemit.endList();   // ranges
  jemit.endObject(); // name
}

/// Prints source location information as debug information.
void printLocation(Location loc, emitter::JsonEmitter &jemit) {
  jemit.startNamedObject("debuginfo");
  jemit.printKVPair("type", "DebugInfo");

  std::string location;
  llvm::raw_string_ostream locationStream(location);
  loc.print(locationStream);

  location.erase(0, location.find('"') + 1);
  location.erase(location.length() - 1, 1);

  std::string fileName = location.substr(0, location.find('"'));
  location.erase(0, location.find('"') + 2);

  std::string line = location.substr(0, location.find(":"));
  location.erase(0, location.find(":") + 1);

  std::string col = location.substr(0, location.find(":"));
  location.erase(0, location.find(":") + 1);

  jemit.printKVPair("start_line", line, /*stringify=*/false);
  jemit.printKVPair("end_line", line, /*stringify=*/false);
  jemit.printKVPair("start_column", col, /*stringify=*/false);
  jemit.printKVPair("end_column", col, /*stringify=*/false);
  jemit.printKVPair("filename", fileName);

  jemit.endObject(); // debuginfo
}

//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//

/// Emits this array to the output stream.
void Array::emit(emitter::JsonEmitter &jemit) {
  jemit.startNamedObject(name);

  // FIXME: Rewrite to check with sdfg args instead of string
  bool isArg = false;
  if (name.find("arg") != std::string::npos) {
    isArg = true;
  }

  if (stream) {
    jemit.printKVPair("type", "Stream");
  } else if (shape.getShape().empty() && !isArg) {
    jemit.printKVPair("type", "Scalar");
  } else {
    jemit.printKVPair("type", "Array");
  }

  jemit.startNamedObject("attributes");

  jemit.printKVPair("transient", transient ? "true" : "false",
                    /*stringify=*/false);

  jemit.printKVPair("dtype",
                    dtypeToString(typeToDtype(shape.getElementType())));

  jemit.startNamedList("shape");

  if (shape.getShape().empty()) {
    jemit.startEntry();
    jemit.printString("1");
  }

  unsigned intIdx = 0;
  unsigned symIdx = 0;
  SmallVector<std::string> strideList = {"1"};
  unsigned intStrideIdx = shape.getIntegers().size() - 1;
  unsigned symStrideIdx = shape.getSymbols().size() - 1;

  for (unsigned i = 0; i < shape.getShape().size(); ++i) {
    jemit.startEntry();
    if (shape.getShape()[i]) {
      jemit.printString(std::to_string(shape.getIntegers()[intIdx]));

      if (i > 0) {
        std::string newString =
            strideList.back() + " * " +
            std::to_string(shape.getIntegers()[intStrideIdx]);
        strideList.push_back(newString);
        intStrideIdx--;
      }
      ++intIdx;
    } else {
      jemit.printString(shape.getSymbols()[symIdx].str());

      if (i > 0) {
        std::string newString =
            strideList.back() + " * " + shape.getSymbols()[symStrideIdx].str();
        strideList.push_back(newString);
        symStrideIdx--;
      }
      ++symIdx;
    }
  }

  jemit.endList(); // shape

  if (!shape.getShape().empty()) {
    jemit.startNamedList("strides");

    for (int i = strideList.size() - 1; i >= 0; --i) {
      jemit.startEntry();
      jemit.printString(strideList[i]);
    }

    jemit.endList(); // strides
  }

  jemit.endObject(); // attributes
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

/// Emits this range to the output stream.
void translation::Range::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("start", start);
  jemit.printKVPair("end", end);
  jemit.printKVPair("step", step);
  jemit.printKVPair("tile", tile);
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// InterstateEdge
//===----------------------------------------------------------------------===//

/// Sets the condition of the interstate edge.
void InterstateEdge::setCondition(Condition condition) {
  ptr->setCondition(condition);
}

/// Adds an assignment to the interstate edge.
void InterstateEdge::addAssignment(Assignment assignment) {
  ptr->addAssignment(assignment);
}

/// Emits the interstate edge to the output stream.
void InterstateEdge::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Sets the condition of the interstate edge.
void InterstateEdgeImpl::setCondition(Condition condition) {
  this->condition = condition;
}

/// Adds an assignment to the interstate edge.
void InterstateEdgeImpl::addAssignment(Assignment assignment) {
  assignments.push_back(assignment);
}

/// Emits the interstate edge to the output stream.
void InterstateEdgeImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Edge");
  jemit.printKVPair("src", source.getID());
  jemit.printKVPair("dst", destination.getID());

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "InterstateEdge");

  jemit.startNamedObject("attributes");

  jemit.startNamedObject("assignments");
  for (Assignment a : assignments)
    jemit.printKVPair(a.key, a.value);
  jemit.endObject(); // assignments

  jemit.startNamedObject("condition");
  jemit.printKVPair("string_data", condition.condition);
  jemit.printKVPair("language", "Python");
  jemit.endObject(); // condition

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// MultiEdge
//===----------------------------------------------------------------------===//

/// Returns the source connector of this edge.
Connector MultiEdge::getSource() { return source; }

/// Returns the destination connector of this edge.
Connector MultiEdge::getDestination() { return destination; }

/// Makes this edge a dependency edge.
void MultiEdge::makeDependence() { depEdge = true; }

/// Emits this edge to the output stream.
void MultiEdge::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MultiConnectorEdge");

  jemit.printKVPair("src", source.parent.getID());
  jemit.printKVPair("dst", destination.parent.getID());

  jemit.printKVPair("src_connector", depEdge ? "null" : source.name,
                    /*stringify=*/!source.isNull && !depEdge);

  jemit.printKVPair("dst_connector", depEdge ? "null" : destination.name,
                    /*stringify=*/!destination.isNull && !depEdge);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.startNamedObject("data");
  jemit.printKVPair("type", "Memlet");
  jemit.startNamedObject("attributes");

  if (!source.data.empty() && !depEdge) {
    jemit.printKVPair("data", source.data);
  } else if (!destination.data.empty() && !depEdge) {
    jemit.printKVPair("data", destination.data);
  }

  printRangeVector(source.ranges, "subset", jemit);
  printRangeVector(source.ranges, "src_subset", jemit);

  printRangeVector(destination.ranges, "other_subset", jemit);
  printRangeVector(destination.ranges, "dst_subset", jemit);

  jemit.endObject(); // attributes
  jemit.endObject(); // data
  jemit.endObject(); // attributes

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

/// Sets the ID of the Node.
void Node::setID(unsigned id) { ptr->setID(id); }

/// Returns the ID of the Node.
unsigned Node::getID() { return ptr->getID(); }

/// Returns the source code location.
Location Node::getLocation() { return ptr->getLocation(); }

/// Returns the type of the Node.
NType Node::getType() { return type; }

/// Sets the name of the node.
void Node::setName(StringRef name) { ptr->setName(name); }

/// Returns the name of the node.
StringRef Node::getName() { return ptr->getName(); }

/// Sets the parent of the node.
void Node::setParent(Node parent) { ptr->setParent(parent); }

/// Returns the parent of the node.
Node Node::getParent() { return ptr->getParent(); }

/// Return true if this node has a parent node.
bool Node::hasParent() { return getParent().ptr != nullptr; }

/// Returns the top-level SDFG.
SDFG Node::getSDFG() {
  if (type == NType::SDFG) {
    return SDFG(std::static_pointer_cast<SDFGImpl>(ptr));
  }
  return ptr->getParent().getSDFG();
}

/// Returns the surrounding state.
State Node::getState() {
  if (type == NType::State) {
    return State(std::static_pointer_cast<StateImpl>(ptr));
  }
  return ptr->getParent().getState();
}

/// Adds an attribute to this node, replaces existing attributes with the same
/// name.
void Node::addAttribute(Attribute attribute) { ptr->addAttribute(attribute); }

/// Emits this node to the output stream.
void Node::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Sets the ID of the node.
void NodeImpl::setID(unsigned id) { this->id = id; }

/// Returns the ID of the node.
unsigned NodeImpl::getID() { return id; }

/// Returns the source code location.
Location NodeImpl::getLocation() { return location; }

/// Sets the name of the node.
void NodeImpl::setName(StringRef name) {
  this->name = name.str();
  utils::sanitizeName(this->name);
}

/// Returns the name of the node.
StringRef NodeImpl::getName() { return name; }

/// Sets the parent of the node.
void NodeImpl::setParent(Node parent) { this->parent = parent; }

/// Returns the parent of the node.
Node NodeImpl::getParent() { return parent; }

/// Adds an attribute to this node, replaces existing attributes with the same
/// name.
void NodeImpl::addAttribute(Attribute attribute) {
  attributes.push_back(attribute);
}

/// Emits this node to the output stream.
void NodeImpl::emit(emitter::JsonEmitter &jemit) {}

//===----------------------------------------------------------------------===//
// ConnectorNode
//===----------------------------------------------------------------------===//

/// Adds an incoming connector.
void ConnectorNode::addInConnector(Connector connector) {
  ptr->addInConnector(connector);
}

/// Adds an outgoing connector.
void ConnectorNode::addOutConnector(Connector connector) {
  ptr->addOutConnector(connector);
}

/// Returns to number of incoming connectors.
unsigned ConnectorNode::getInConnectorCount() {
  return ptr->getInConnectorCount();
}

/// Returns to number of outgoing connectors.
unsigned ConnectorNode::getOutConnectorCount() {
  return ptr->getOutConnectorCount();
}

/// Emits the connectors to the output stream.
void ConnectorNode::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Adds an incoming connector.
void ConnectorNodeImpl::addInConnector(Connector connector) {
  for (Connector c : inConnectors)
    if (c.name == connector.name && c.parent == connector.parent) {
      if (c == connector) {
        return;
      } else {
        emitError(location, "Conflicting duplicate connectors in "
                            "ConnectorNodeImpl::addInConnector: " +
                                c.name);
      }
    }

  inConnectors.push_back(connector);
}

/// Adds an outgoing connector.
void ConnectorNodeImpl::addOutConnector(Connector connector) {
  for (Connector c : outConnectors)
    if (c.name == connector.name && c.parent == connector.parent) {
      if (c == connector) {
        return;
      } else {
        emitError(location, "Conflicting duplicate connectors in "
                            "ConnectorNodeImpl::addOutConnector: " +
                                c.name);
      }
    }

  outConnectors.push_back(connector);
}

/// Returns to number of incoming connectors.
unsigned ConnectorNodeImpl::getInConnectorCount() {
  return inConnectors.size();
}

/// Returns to number of outgoing connectors.
unsigned ConnectorNodeImpl::getOutConnectorCount() {
  return outConnectors.size();
}

/// Emits the connectors to the output stream.
void ConnectorNodeImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startNamedObject("in_connectors");
  for (Connector c : inConnectors) {
    if (c.isNull)
      continue;
    jemit.printKVPair(c.name, "null", /*stringify=*/false);
  }
  jemit.endObject(); // in_connectors

  jemit.startNamedObject("out_connectors");
  for (Connector c : outConnectors) {
    if (c.isNull)
      continue;
    jemit.printKVPair(c.name, "null", /*stringify=*/false);
  }
  jemit.endObject(); // out_connectors
}

//===----------------------------------------------------------------------===//
// ScopeNode
//===----------------------------------------------------------------------===//

/// Adds a connector node to the scope.
void ScopeNode::addNode(ConnectorNode node) {
  if (!node.hasParent()) {
    node.setParent(*this);
  }
  ptr->addNode(node);
}

/// Adds a multiedge from the source to the destination connector.
void ScopeNode::routeWrite(Connector from, Connector to, Value mapValue) {
  ptr->routeWrite(from, to, mapValue);
}

/// Adds an edge to the scope.
void ScopeNode::addEdge(MultiEdge edge) { ptr->addEdge(edge); }

/// Maps the MLIR value to the specified connector.
void ScopeNode::mapConnector(Value value, Connector connector) {
  ptr->mapConnector(value, connector);
}

/// Returns the connector associated with a MLIR value.
Connector ScopeNode::lookup(Value value) {
  if (type == NType::MapEntry) {
    return MapEntry(*this).lookup(value);
  }

  if (type == NType::ConsumeEntry) {
    return ConsumeEntry(*this).lookup(value);
  }

  return ptr->lookup(value);
}

/// Adds a dependency edge between the MLIR and the connector.
void ScopeNode::addDependency(Value value, Connector connector) {
  if (type == NType::MapEntry) {
    return MapEntry(*this).addDependency(value, connector);
  }

  if (type == NType::ConsumeEntry) {
    return ConsumeEntry(*this).addDependency(value, connector);
  }

  ptr->addDependency(value, connector);
}

/// Emits all nodes and edges to the output stream.
void ScopeNode::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Adds a connector node to the scope.
void ScopeNodeImpl::addNode(ConnectorNode node) {
  node.setID(nodes.size());
  nodes.push_back(node);
}

/// Adds a multiedge from the source to the destination connector.
void ScopeNodeImpl::routeWrite(Connector from, Connector to, Value mapValue) {
  addNode(to.parent);
  addEdge(MultiEdge(location, from, to));

  Connector out(to.parent);
  out.setData(to.data);
  to.parent.addOutConnector(out);
  mapConnector(mapValue, out);
}

/// Adds an edge to the scope.
void ScopeNodeImpl::addEdge(MultiEdge edge) { edges.push_back(edge); }

/// Maps the MLIR value to the specified connector.
void ScopeNodeImpl::mapConnector(Value value, Connector connector) {
  auto res = lut.insert({utils::valueToString(value), connector});

  if (!res.second)
    res.first->second = connector;
}

/// Returns the connector associated with a MLIR value.
Connector ScopeNodeImpl::lookup(Value value) {
  if (lut.find(utils::valueToString(value)) == lut.end()) {
    emitError(location,
              "Tried to lookup nonexistent value in ScopeNodeImpl::lookup");
  }
  return lut.find(utils::valueToString(value))->second;
}

/// Adds a dependency edge between the MLIR and the connector.
void ScopeNodeImpl::addDependency(Value value, Connector connector) {
  MultiEdge edge(location, lookup(value), connector);
  edge.makeDependence();
  addEdge(edge);
}

/// Emits all nodes and edges to the output stream.
void ScopeNodeImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startNamedList("nodes");
  for (ConnectorNode cn : nodes)
    cn.emit(jemit);
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  for (MultiEdge me : edges)
    me.emit(jemit);
  jemit.endList(); // edges
}

//===----------------------------------------------------------------------===//
// SDFG
//===----------------------------------------------------------------------===//

/// Returns the state associated with the provided name.
State SDFG::lookup(StringRef name) { return ptr->lookup(name); }

/// Adds a state to the SDFG.
void SDFG::addState(State state) { ptr->addState(state); }

/// Adds a state to the SDFG and marks it as the entry state.
void SDFG::setStartState(State state) { ptr->setStartState(state); }

/// Adds an interstate edge to the SDFG, connecting two states.
void SDFG::addEdge(InterstateEdge edge) { ptr->addEdge(edge); }

/// Adds an array (data container) to the SDFG.
void SDFG::addArray(Array array) { ptr->addArray(array); }

/// Adds an array (data container) to the SDFG and marks it as an argument.
void SDFG::addArg(Array arg) { ptr->addArg(arg); }

/// Adds a symbol to the SDFG.
void SDFG::addSymbol(Symbol symbol) { ptr->addSymbol(symbol); }

/// Returns an array of all symbols in the SDFG.
std::vector<Symbol> SDFG::getSymbols() { return ptr->getSymbols(); }

/// Sets all non-argument arrays to transient.
void SDFG::setNestedTransient() { ptr->setNestedTransient(); }

/// Emits the SDFG to the output stream.
void SDFG::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); };

/// Emits the SDFG as a nested SDFG to the output stream.
void SDFG::emitNested(emitter::JsonEmitter &jemit) { ptr->emitNested(jemit); };

/// Global counter for the ID of SDFGs.
unsigned SDFGImpl::list_id = 0;

/// Returns the state associated with the provided name.
State SDFGImpl::lookup(StringRef name) { return lut.find(name.str())->second; }

/// Adds a state to the SDFG.
void SDFGImpl::addState(State state) {
  state.setParent(SDFG(shared_from_this()));
  state.setID(states.size());
  states.push_back(state);

  if (!lut.insert({state.getName().str(), state}).second)
    emitError(location, "Duplicate ID in SDFGImpl::addState");
}

/// Adds a state to the SDFG and marks it as the entry state.
void SDFGImpl::setStartState(State state) {
  if (std::find(states.begin(), states.end(), state) == states.end())
    emitError(
        location,
        "Non-existent state assigned as start in SDFGImpl::setStartState");
  else
    this->startState = state;
}

/// Adds an interstate edge to the SDFG, connecting two states.
void SDFGImpl::addEdge(InterstateEdge edge) { edges.push_back(edge); }

/// Adds an array (data container) to the SDFG.
void SDFGImpl::addArray(Array array) { arrays.push_back(array); }

/// Adds an array (data container) to the SDFG and marks it as an argument.
void SDFGImpl::addArg(Array arg) {
  args.push_back(arg);
  addArray(arg);
}

/// Adds a symbol to the SDFG.
void SDFGImpl::addSymbol(Symbol symbol) { symbols.push_back(symbol); }

/// Returns an array of all symbols in the SDFG.
std::vector<Symbol> SDFGImpl::getSymbols() { return symbols; }

/// Sets all non-argument arrays to transient.
void SDFGImpl::setNestedTransient() {
  for (Array &a : arrays)
    if (!llvm::is_contained(args, a))
      a.transient = true;
}

/// Emits the SDFG to the output stream.
void SDFGImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  emitBody(jemit);
}

/// Emits the SDFG as a nested SDFG to the output stream.
void SDFGImpl::emitNested(emitter::JsonEmitter &jemit) {
  jemit.startNamedObject("sdfg");
  emitBody(jemit);
}

/// Emits the body of the SDFG to the output stream.
void SDFGImpl::emitBody(emitter::JsonEmitter &jemit) {
  jemit.printKVPair("type", "SDFG");
  jemit.printKVPair("sdfg_list_id", id, /*stringify=*/false);
  jemit.printKVPair("start_state", startState.getID(),
                    /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.printKVPair("name", name);

  jemit.startNamedList("arg_names");
  for (Array a : args) {
    jemit.startEntry();
    jemit.printString(a.name);
  }
  jemit.endList(); // arg_names

  jemit.startNamedObject("constants_prop");
  jemit.endObject(); // constants_prop

  jemit.startNamedObject("_arrays");
  for (Array a : arrays)
    a.emit(jemit);
  jemit.endObject(); // _arrays

  jemit.startNamedObject("symbols");
  for (Symbol s : symbols)
    jemit.printKVPair(s.name, dtypeToString(s.type));
  jemit.endObject(); // symbols

  jemit.endObject(); // attributes

  jemit.startNamedList("nodes");
  for (State s : states)
    s.emit(jemit);
  jemit.endList(); // nodes

  jemit.startNamedList("edges");
  for (InterstateEdge e : edges)
    e.emit(jemit);
  jemit.endList(); // edges

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// NestedSDFG
//===----------------------------------------------------------------------===//

/// Emits the nested SDFG to the output stream.
void NestedSDFG::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Emits the nested SDFG to the output stream.
void NestedSDFGImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "NestedSDFG");
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.printKVPair("label", name);
  ConnectorNodeImpl::emit(jemit);

  jemit.startNamedObject("symbol_mapping");
  for (Symbol s : parent.getSDFG().getSymbols()) {
    jemit.printKVPair(s.name, s.name);
    sdfg.addSymbol(s);
  }
  jemit.endObject(); // symbol_mapping

  sdfg.emitNested(jemit);

  jemit.endObject(); // attributes
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

/// Modified lookup function creates access nodes if the value could not be
/// found.
Connector State::lookup(Value value) { return ptr->lookup(value); }

/// Emits the state node to the output stream.
void State::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Modified lookup function creates access nodes if the value could not be
/// found.
Connector StateImpl::lookup(Value value) {
  if (lut.find(utils::valueToString(value)) == lut.end()) {
    std::string name = utils::valueToString(value);
    bool init = false;

    if (value.getDefiningOp() != nullptr) {
      AllocOp allocOp = cast<AllocOp>(value.getDefiningOp());
      name = allocOp.getName().value_or(name);
      init = allocOp->hasAttr("init");
    }

    Access access(location, init);
    access.setName(name);
    addNode(access);

    Connector accOut(access);
    accOut.setData(name);
    access.addOutConnector(accOut);

    return accOut;
  }

  return ScopeNodeImpl::lookup(value);
}

/// Emits the state node to the output stream.
void StateImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "SDFGState");
  printLocation(location, jemit);
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  jemit.endObject(); // attributes

  ScopeNodeImpl::emit(jemit);
  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Tasklet
//===----------------------------------------------------------------------===//

/// Sets the code of the tasklet.
void Tasklet::setCode(Code code) { ptr->setCode(code); }

/// Sets the global code of the tasklet.
void Tasklet::setGlobalCode(Code code_global) {
  ptr->setGlobalCode(code_global);
}

/// Sets the side effect flag of the tasklet.
void Tasklet::setHasSideEffect(bool hasSideEffect) {
  ptr->setHasSideEffect(hasSideEffect);
}

/// Emits the tasklet to the output stream.
void Tasklet::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Sets the code of the tasklet.
void TaskletImpl::setCode(Code code) { this->code = code; }

/// Sets the global code of the tasklet.
void TaskletImpl::setGlobalCode(Code code_global) {
  this->code_global = code_global;
}

/// Sets the side effect flag of the tasklet.
void TaskletImpl::setHasSideEffect(bool hasSideEffect) {
  this->hasSideEffect = hasSideEffect;
}

/// Emits the tasklet to the output stream.
void TaskletImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "Tasklet");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.printKVPair("label", name);

  jemit.startNamedObject("code");
  jemit.printKVPair("string_data", code.data);
  jemit.printKVPair("language", codeLanguageToString(code.language));
  jemit.endObject(); // code

  jemit.startNamedObject("code_global");
  jemit.printKVPair("string_data", code_global.data);
  jemit.printKVPair("language", codeLanguageToString(code_global.language));
  jemit.endObject(); // code_global

  jemit.printKVPair("side_effects", hasSideEffect ? "true" : "false",
                    /*stringify=*/false);
  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Library
//===----------------------------------------------------------------------===//

/// Sets the library code path.
void Library::setClasspath(StringRef classpath) {
  ptr->setClasspath(classpath);
}

/// Emits the library node to the output stream.
void Library::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Sets the library code path.
void LibraryImpl::setClasspath(StringRef classpath) {
  this->classpath = classpath.str();
}

/// Emits the library node to the output stream.
void LibraryImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "LibraryNode");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);
  jemit.printKVPair("classpath", classpath);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.printKVPair("name", name);
  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Access
//===----------------------------------------------------------------------===//

/// Returns true if this access node should initialize.
bool Access::getInit() { return ptr->getInit(); }

/// Emits the access node to the output stream
void Access::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Returns true if this access node should initialize.
bool AccessImpl::getInit() { return init; }

/// Emits the access node to the output stream
void AccessImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "AccessNode");
  jemit.printKVPair("label", name);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.printKVPair("data", name);

  jemit.printKVPair("setzero", init ? "true" : "false",
                    /*stringify=*/false);
  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//

/// Adds a parameter to the map entry.
void MapEntry::addParam(StringRef param) { ptr->addParam(param); }

/// Adds a range for a parameter.
void MapEntry::addRange(Range range) { ptr->addRange(range); }

/// Sets the map exit this map entry belongs to.
void MapEntry::setExit(MapExit exit) { ptr->setExit(exit); }

/// Returns the matching map exit.
MapExit MapEntry::getExit() { return ptr->getExit(); }

/// Connects dangling nodes to the map entry.
void MapEntry::connectDanglingNodes() { return ptr->connectDanglingNodes(); }

/// Adds a connector node to the scope.
void MapEntry::addNode(ConnectorNode node) { ptr->addNode(node); }

/// Adds a multiedge from the source to the destination connector.
void MapEntry::routeWrite(Connector from, Connector to, Value mapValue) {
  ptr->routeWrite(from, to, mapValue);
}

/// Adds an edge to the scope.
void MapEntry::addEdge(MultiEdge edge) { ptr->addEdge(edge); }

/// Maps the MLIR value to the specified connector.
void MapEntry::mapConnector(Value value, Connector connector) {
  ptr->mapConnector(value, connector);
}

/// Returns the connector associated with a MLIR value, inserting map
/// connectors when needed.
Connector MapEntry::lookup(Value value) { return ptr->lookup(value); }

/// Adds a dependency edge between the MLIR and the connector.
void MapEntry::addDependency(Value value, Connector connector) {
  ptr->addDependency(value, connector);
}

/// Emits the map entry to the output stream.
void MapEntry::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Adds a parameter to the map entry.
void MapEntryImpl::addParam(StringRef param) { params.push_back(param.str()); }

/// Adds a range for a parameter.
void MapEntryImpl::addRange(Range range) { ranges.push_back(range); }

/// Sets the map exit this map entry belongs to.
void MapEntryImpl::setExit(MapExit exit) { this->exit = exit; }

/// Returns the matching map exit.
MapExit MapEntryImpl::getExit() { return exit; }

/// Connects dangling nodes to the map entry.
void MapEntryImpl::connectDanglingNodes() {
  Connector mapEntryNull(getExit().getEntry());
  addOutConnector(mapEntryNull);
  Connector mapExitNull(getExit());
  getExit().addOutConnector(mapExitNull);

  // Ensure map entry and exit are connected.
  MultiEdge edge(location, mapEntryNull, mapExitNull);
  addEdge(edge);

  // Connect all pending writes that are not read in this map.
  for (const auto &[from, to, mapValue] : writeQueue) {
    bool skip = false;
    for (MultiEdge edge : edges)
      skip |= edge.getSource().parent == from.parent;
    if (skip)
      continue;

    routeOut(from, to, mapValue);
  }

  // Connect all nodes without an ingoing edge.
  for (ConnectorNode node : nodes) {
    bool skip = false;
    for (MultiEdge edge : edges)
      skip |= edge.getDestination().parent == node;
    if (skip || node.getType() == NType::ConsumeExit ||
        node.getType() == NType::MapExit)
      continue;

    Connector connector(node);
    node.addInConnector(node);
    MultiEdge edge(location, mapEntryNull, connector);
    addEdge(edge);
  }

  // Connect all nodes without an outgoing edge.
  for (ConnectorNode node : nodes) {
    bool skip = false;
    for (MultiEdge edge : edges)
      skip |= edge.getSource().parent == node;
    if (skip || node.getType() == NType::ConsumeEntry ||
        node.getType() == NType::MapEntry)
      continue;

    Connector connector(node);
    node.addOutConnector(node);
    MultiEdge edge(location, connector, mapExitNull);
    addEdge(edge);
  }
}

/// Adds a connector node to the scope.
void MapEntryImpl::addNode(ConnectorNode node) {
  node.setParent(MapEntry(shared_from_this()));
  parent.getState().addNode(node);
  nodes.push_back(node);
}

/// Routes the write to the outer scope.
void MapEntryImpl::routeOut(Connector from, Connector to, Value mapValue) {
  MapExit mapExit = getExit();
  Connector in(mapExit, "IN_" + utils::valueToString(mapValue));
  in.setData(from.data);
  mapExit.addInConnector(in);
  addEdge(MultiEdge(location, from, in));

  Connector out(mapExit, "OUT_" + utils::valueToString(mapValue));
  out.setData(from.data);
  out.setRanges(from.ranges);
  mapExit.addOutConnector(out);

  ScopeNode scope(parent);
  scope.routeWrite(out, to, mapValue);
}

/// Adds a multiedge from the source to the destination connector.
void MapEntryImpl::routeWrite(Connector from, Connector to, Value mapValue) {
  Access toAcc(to.parent);

  Access access(location, toAcc.getInit());
  access.setName(toAcc.getName());

  Connector accIn(access);
  accIn.setData(to.data);
  accIn.setRanges(to.ranges);
  access.addInConnector(accIn);

  addNode(access);
  addEdge(MultiEdge(location, from, accIn));

  Connector accOut(access);
  accOut.setData(to.data);
  access.addOutConnector(accOut);
  mapConnector(mapValue, accOut);

  to.setRanges({});
  writeQueue.push_back(std::make_tuple(accOut, to, mapValue));
}

/// Adds an edge to the scope.
void MapEntryImpl::addEdge(MultiEdge edge) {
  parent.getState().addEdge(edge);
  edges.push_back(edge);
}

/// Maps the MLIR value to the specified connector.
void MapEntryImpl::mapConnector(Value value, Connector connector) {
  auto res = lut.insert({utils::valueToString(value), connector});

  if (!res.second)
    res.first->second = connector;
}

/// Returns the connector associated with a MLIR value, inserting map
/// connectors when needed.
Connector MapEntryImpl::lookup(Value value) {
  if (lut.find(utils::valueToString(value)) == lut.end()) {
    ScopeNode scope(parent);
    Connector srcConn = scope.lookup(value);

    MapEntry mapEntry(shared_from_this());
    Connector dstConn(mapEntry, "IN_" + utils::valueToString(value));
    dstConn.setData(srcConn.data);
    addInConnector(dstConn);

    MultiEdge multiedge(location, srcConn, dstConn);
    scope.addEdge(multiedge);

    Connector outConn(mapEntry, "OUT_" + utils::valueToString(value));
    outConn.setData(srcConn.data);
    outConn.setRanges(srcConn.ranges);
    addOutConnector(outConn);
    mapConnector(value, outConn);
  }

  return ScopeNodeImpl::lookup(value);
}

/// Adds a dependency edge between the MLIR and the connector.
void MapEntryImpl::addDependency(Value value, Connector connector) {
  if (lut.find(utils::valueToString(value)) == lut.end()) {
    MapEntry entry(shared_from_this());
    Connector mapIn(entry);
    addInConnector(mapIn);
    Connector mapOut(entry);
    addOutConnector(mapOut);

    MultiEdge edge(location, mapOut, connector);
    edge.makeDependence();
    addEdge(edge);

    ScopeNode scope(parent);
    scope.addDependency(value, mapIn);
    return;
  }

  MultiEdge edge(location, lookup(value), connector);
  edge.makeDependence();
  addEdge(edge);
}

/// Emits the map entry to the output stream.
void MapEntryImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MapEntry");
  jemit.printKVPair("label", getName());
  jemit.printKVPair("scope_exit", exit.getID());
  jemit.printKVPair("id", getID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.printKVPair("label", getName());

  jemit.startNamedList("params");
  for (std::string s : params) {
    jemit.startEntry();
    jemit.printString(s);
  }
  jemit.endList(); // params

  printRangeVector(ranges, "range", jemit);

  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}

/// Sets the map entry this map exit belongs to.
void MapExit::setEntry(MapEntry entry) { ptr->setEntry(entry); }

/// Returns the matching map entry.
MapEntry MapExit::getEntry() { return ptr->getEntry(); }

/// Emits the map exit to the output stream.
void MapExit::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Sets the map entry this map exit belongs to.
void MapExitImpl::setEntry(MapEntry entry) { this->entry = entry; }

/// Returns the matching map entry.
MapEntry MapExitImpl::getEntry() { return entry; }

/// Emits the map exit to the output stream.
void MapExitImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "MapExit");
  jemit.printKVPair("label", name);
  jemit.printKVPair("scope_entry", entry.getID());
  jemit.printKVPair("scope_exit", id);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}

//===----------------------------------------------------------------------===//
// Consume
//===----------------------------------------------------------------------===//

/// Sets the consume exit this consume entry belongs to.
void ConsumeEntry::setExit(ConsumeExit exit) { ptr->setExit(exit); }

/// Returns the matching consume exit.
ConsumeExit ConsumeEntry::getExit() { return ptr->getExit(); }

/// Adds a connector node to the scope.
void ConsumeEntry::addNode(ConnectorNode node) { ptr->addNode(node); }

/// Adds a multiedge from the source to the destination connector.
void ConsumeEntry::routeWrite(Connector from, Connector to, Value mapValue) {
  ptr->routeWrite(from, to, mapValue);
}

/// Adds an edge to the scope.
void ConsumeEntry::addEdge(MultiEdge edge) { ptr->addEdge(edge); }

/// Maps the MLIR value to the specified connector.
void ConsumeEntry::mapConnector(Value value, Connector connector) {
  ptr->mapConnector(value, connector);
}

/// Returns the connector associated with a MLIR value, inserting consume
/// connectors when needed.
Connector ConsumeEntry::lookup(Value value) { return ptr->lookup(value); }

/// Adds a dependency edge between the MLIR and the connector.
void ConsumeEntry::addDependency(Value value, Connector connector) {
  ptr->addDependency(value, connector);
}

/// Sets the number of processing elements.
void ConsumeEntry::setNumPes(StringRef pes) { ptr->setNumPes(pes); }

/// Sets the name of the processing element index.
void ConsumeEntry::setPeIndex(StringRef pe) { ptr->setPeIndex(pe); }

/// Sets the condition to continue stream consumption.
void ConsumeEntry::setCondition(Code condition) {
  ptr->setCondition(condition);
}

/// Emits the consume entry to the output stream.
void ConsumeEntry::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Sets the consume exit this consume entry belongs to.
void ConsumeEntryImpl::setExit(ConsumeExit exit) { this->exit = exit; }

/// Returns the matching consume exit.
ConsumeExit ConsumeEntryImpl::getExit() { return exit; }

/// Adds a connector node to the scope.
void ConsumeEntryImpl::addNode(ConnectorNode node) {
  node.setParent(ConsumeEntry(shared_from_this()));
  getParent().getState().addNode(node);
}

/// Adds a multiedge from the source to the destination connector.
void ConsumeEntryImpl::routeWrite(Connector from, Connector to,
                                  Value mapValue) {
  // FIXME: Should be fixed like map entry (including dependency routing)
  ConsumeExit consumeExit = getExit();
  Connector in(consumeExit,
               "IN_" + std::to_string(consumeExit.getInConnectorCount()));
  in.setData(from.data);
  in.setRanges(from.ranges);
  consumeExit.addInConnector(in);

  MultiEdge edge(location, from, in);
  addEdge(edge);

  Connector out(consumeExit,
                "OUT_" + std::to_string(consumeExit.getOutConnectorCount()));
  out.setData(in.data);
  out.setRanges(in.ranges);
  consumeExit.addOutConnector(out);

  ScopeNode scope(parent);
  scope.routeWrite(out, to, mapValue);
}

/// Adds an edge to the scope.
void ConsumeEntryImpl::addEdge(MultiEdge edge) {
  getParent().getState().addEdge(edge);
}

/// Maps the MLIR value to the specified connector.
void ConsumeEntryImpl::mapConnector(Value value, Connector connector) {
  auto res = lut.insert({utils::valueToString(value), connector});

  if (!res.second)
    res.first->second = connector;
}

/// Returns the connector associated with a MLIR value, inserting consume
/// connectors when needed.
Connector ConsumeEntryImpl::lookup(Value value) {
  if (lut.find(utils::valueToString(value)) == lut.end()) {
    ScopeNode scope(parent);
    ConsumeEntry entry(shared_from_this());

    Connector srcConn = scope.lookup(value);
    Connector dstConn(entry, "IN_" + utils::valueToString(value));
    dstConn.setData(srcConn.data);
    dstConn.setRanges(srcConn.ranges);
    addInConnector(dstConn);

    MultiEdge multiedge(location, srcConn, dstConn);
    scope.addEdge(multiedge);

    Connector outConn(entry, "OUT_" + utils::valueToString(value));
    outConn.setData(dstConn.data);
    outConn.setRanges(dstConn.ranges);
    addOutConnector(outConn);
    ScopeNodeImpl::mapConnector(value, outConn);
  }

  return ScopeNodeImpl::lookup(value);
}

/// Adds a dependency edge between the MLIR and the connector.
void ConsumeEntryImpl::addDependency(Value value, Connector connector) {
  if (lut.find(utils::valueToString(value)) == lut.end()) {
    ConsumeEntry entry(shared_from_this());
    Connector consIn(entry);
    addInConnector(consIn);
    Connector consOut(entry);
    addOutConnector(consOut);

    MultiEdge edge(location, consOut, connector);
    edge.makeDependence();
    addEdge(edge);

    ScopeNode scope(parent);
    scope.addDependency(value, consIn);
    return;
  }

  MultiEdge edge(location, lookup(value), connector);
  edge.makeDependence();
  addEdge(edge);
}

/// Sets the number of processing elements.
void ConsumeEntryImpl::setNumPes(StringRef pes) { num_pes = pes.str(); }

/// Sets the name of the processing element index.
void ConsumeEntryImpl::setPeIndex(StringRef pe) {
  pe_index = pe.str();
  utils::sanitizeName(pe_index);
}

/// Sets the condition to continue stream consumption.
void ConsumeEntryImpl::setCondition(Code condition) {
  this->condition = condition;
}

/// Emits the consume entry to the output stream.
void ConsumeEntryImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "ConsumeEntry");
  jemit.printKVPair("label", getName());
  jemit.printKVPair("scope_exit", exit.getID());
  jemit.printKVPair("id", getID(), /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  jemit.printKVPair("label", getName());

  if (num_pes.empty()) {
    jemit.printKVPair("num_pes", "null", /*stringify=*/false);
  } else {
    jemit.printKVPair("num_pes", num_pes);
  }

  jemit.printKVPair("pe_index", pe_index);

  jemit.startNamedObject("condition");
  jemit.printKVPair("string_data", condition.data);
  jemit.printKVPair("language", codeLanguageToString(condition.language));
  jemit.endObject(); // condition

  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}

/// Sets the consume entry this consume exit belongs to.
void ConsumeExit::setEntry(ConsumeEntry entry) { ptr->setEntry(entry); }

/// Emits the consume exit to the output stream.
void ConsumeExit::emit(emitter::JsonEmitter &jemit) { ptr->emit(jemit); }

/// Sets the consume entry this consume exit belongs to.
void ConsumeExitImpl::setEntry(ConsumeEntry entry) { this->entry = entry; }

/// Emits the consume exit to the output stream.
void ConsumeExitImpl::emit(emitter::JsonEmitter &jemit) {
  jemit.startObject();
  jemit.printKVPair("type", "ConsumeExit");
  jemit.printKVPair("label", name);
  jemit.printKVPair("scope_entry", entry.getID());
  jemit.printKVPair("scope_exit", id);
  jemit.printKVPair("id", id, /*stringify=*/false);

  jemit.startNamedObject("attributes");
  printLocation(location, jemit);
  ConnectorNodeImpl::emit(jemit);
  jemit.endObject(); // attributes */

  jemit.endObject();
}
