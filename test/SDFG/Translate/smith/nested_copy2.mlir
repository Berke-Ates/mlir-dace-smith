// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module {
  sdfg.sdfg () -> (){
    sdfg.state @state_0{

      sdfg.nested_sdfg () -> (){
        sdfg.state @state_1{
          %6 = sdfg.alloc {init} () : !sdfg.array<f16>
          sdfg.copy %6 -> %6 : !sdfg.array<f16>
        }
      }

    }
  }
}
