// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<2x6xi32>

  sdfg.state @state_0 {
    sdfg.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
    }
  }
}
