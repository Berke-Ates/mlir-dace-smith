// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: size of lower bounds matches size of arguments

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    sdfg.map (%i, %j) = () to () step () {
    }
  }
}
