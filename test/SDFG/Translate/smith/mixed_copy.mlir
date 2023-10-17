// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module {
  sdfg.sdfg {entry = @state_1} () -> (){
    %0 = sdfg.alloc {init} () : !sdfg.array<2xindex>
    %1 = sdfg.alloc {init} () : !sdfg.array<2xindex>

    sdfg.state @state_1{
      sdfg.map (%arg1) = (0) to (1) step (1){
        sdfg.copy %0 -> %0 : !sdfg.array<2xindex>

        sdfg.map (%arg2) = (0) to (1) step (1){
          sdfg.copy %0 -> %1 : !sdfg.array<2xindex>
        }

        sdfg.copy %0 -> %0 : !sdfg.array<2xindex>
      }

      sdfg.copy %0 -> %0 : !sdfg.array<2xindex>
      sdfg.copy %1 -> %1 : !sdfg.array<2xindex>
    }
  }
}
