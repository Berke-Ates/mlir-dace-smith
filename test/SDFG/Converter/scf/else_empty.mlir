// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main(%c0: i32, %c1: i32) {
  %0 = arith.cmpi ne, %c0, %c1 : i32

  scf.if %0 {
    %c2 = arith.constant 1 : i32
  } else {
   
  }

  return
}
