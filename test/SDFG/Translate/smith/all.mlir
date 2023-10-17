// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module {
  sdfg.sdfg {entry = @state_1} () -> (){
    %0 = sdfg.alloc {init} () : !sdfg.array<2xindex>

    sdfg.state @state_1{
      sdfg.map (%arg1) = (0) to (1) step (1){
        %2 = sdfg.load %0[%arg1] : !sdfg.array<2xindex> -> index
        sdfg.store %2, %0[%arg1] : index -> !sdfg.array<2xindex>

        sdfg.copy %0 -> %0 : !sdfg.array<2xindex>

        %3 = sdfg.tasklet () -> (i32){
          %2 = arith.constant 0 : i32
          sdfg.return %2 : i32
        }

        sdfg.map (%arg2) = (0) to (1) step (1){
          %4 = sdfg.load %0[0] : !sdfg.array<2xindex> -> index
          sdfg.store %4, %0[0] : index -> !sdfg.array<2xindex>

          sdfg.copy %0 -> %0 : !sdfg.array<2xindex>

          %5 = sdfg.tasklet () -> (i32){
            %2 = arith.constant 0 : i32
            sdfg.return %2 : i32
          }
        }

        %6 = sdfg.load %0[1] : !sdfg.array<2xindex> -> index
        sdfg.store %6, %0[1] : index -> !sdfg.array<2xindex>
      }

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

      sdfg.copy %0 -> %0 : !sdfg.array<2xindex>
      sdfg.copy %0 -> %0 : !sdfg.array<2xindex>
    }
  }
}
