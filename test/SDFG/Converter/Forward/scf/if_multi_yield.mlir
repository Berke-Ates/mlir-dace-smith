// RUN: sdfg-opt --convert-to-sdfg %s | sdfg-opt
func.func private @main(%c0: i32, %c1: i32) {
  %0 = arith.cmpi ne, %c0, %c1 : i32

  %x, %y = scf.if %0 -> (i32, i32) {
    scf.yield %c0, %c1 : i32, i32
  } else {
    scf.yield %c1, %c0 : i32, i32
  }

  return
}