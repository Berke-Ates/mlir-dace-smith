// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module {
  sdfg.sdfg {entry = @state_1} () -> (){
    %0 = sdfg.alloc {init} () : !sdfg.array<2xindex>

    sdfg.state @state_1{
        %2 = sdfg.load %0[0] : !sdfg.array<2xindex> -> index
        %3 = sdfg.tasklet (%2 : index) -> (index){
          sdfg.return %2 : index
        }
        sdfg.store %3, %0[0] : index -> !sdfg.array<2xindex>

        %6 = sdfg.load %0[1] : !sdfg.array<2xindex> -> index
        %7 = sdfg.tasklet (%6 : index) -> (index){
          sdfg.return %6 : index
        }
        sdfg.store %7, %0[1] : index -> !sdfg.array<2xindex>
    }
  }
}
