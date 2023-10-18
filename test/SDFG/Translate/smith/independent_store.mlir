// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module {
  sdfg.sdfg {entry = @state_1} () -> (){
    %0 = sdfg.alloc {init} () : !sdfg.array<2xindex>
        %1 = sdfg.alloc {init} () : !sdfg.array<2xindex>

    sdfg.state @state_1{
      sdfg.map (%arg1) = (0) to (1) step (1){
        %4 = sdfg.load %0[0] : !sdfg.array<2xindex> -> index
        sdfg.store %4, %0[0] : index -> !sdfg.array<2xindex>

        %3 = sdfg.load %1[0] : !sdfg.array<2xindex> -> index
        sdfg.store %3, %0[0] : index -> !sdfg.array<2xindex>
      }
    }
  }
}
