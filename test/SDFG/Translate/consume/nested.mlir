// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

sdfg.sdfg () -> () {
  %A = sdfg.alloc() : !sdfg.stream<i32>
  %C = sdfg.alloc() : !sdfg.array<6xi32>

  sdfg.state @state_0 {
    sdfg.consume{num_pes=5} (%A : !sdfg.stream<i32>) -> (pe: %pe, elem: %el) {
      sdfg.consume{num_pes=5} (%A : !sdfg.stream<i32>) -> (pe: %p, elem: %e) {
        %res = sdfg.tasklet(%el: i32, %e: i32) -> (i32) {
          %1 = arith.constant 1 : i32
          %res = arith.addi %el, %e : i32
          sdfg.return %res : i32
        }

        %0 = sdfg.tasklet() -> (index) {
          %0 = arith.constant 0 : index
          sdfg.return %0 : index
        }
        sdfg.store{wcr="add"} %res, %C[%0] : i32 -> !sdfg.array<6xi32>
      }
    }
  }
}